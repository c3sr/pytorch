package predictor

import (
	"bufio"
	"context"
	"io"
	"os"
	"strings"

	"github.com/c3sr/config"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/agent"
	"github.com/c3sr/dlframework/framework/options"
	common "github.com/c3sr/dlframework/framework/predictor"
	"github.com/c3sr/downloadmanager"
	gopytorch "github.com/c3sr/go-pytorch"
	"github.com/c3sr/pytorch"
	"github.com/c3sr/tracer"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	gotensor "gorgonia.org/tensor"
)

// SemanticSegmentationPredictor ...
type SemanticSegmentationPredictor struct {
	common.ImagePredictor
	predictor *gopytorch.Predictor
	labels    []string
}

// NewSemanticSegmentationPredictor ...
func NewSemanticSegmentationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(SemanticSegmentationPredictor)

	return predictor.Load(ctx, model, opts...)
}

// Download ...
func (p *SemanticSegmentationPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &SemanticSegmentationPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

// Load ...
func (p *SemanticSegmentationPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &SemanticSegmentationPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *SemanticSegmentationPredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download graph"),
		)
		checksum := p.GetGraphChecksum()
		if checksum != "" {
			if _, _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
				return err
			}
		}
	}

	span.LogFields(
		olog.String("event", "download features"),
	)
	checksum := p.GetFeaturesChecksum()
	if checksum != "" {
		if _, _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
			return err
		}
	} else {
		if _, _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
			return err
		}
	}

	return nil
}

func (p *SemanticSegmentationPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "read features"),
	)

	var labels []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}
	p.labels = labels

	span.LogFields(
		olog.String("event", "creating predictor"),
	)

	opts, err := p.GetPredictionOptions()
	if err != nil {
		return err
	}

	pred, err := gopytorch.New(
		ctx,
		options.WithOptions(opts),
		options.Graph([]byte(p.GetGraphPath())),
	)
	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

// GetInputLayerName ...
func (p *SemanticSegmentationPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelInputs := model.GetInputs()
	typeParameters := modelInputs[0].GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		// TODO get input layer name => for what..?
		return "", errors.New("cannot determine the name of the input layer")
	}
	return name, nil
}

// GetOutputLayerName ...
func (p *SemanticSegmentationPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelOutput := model.GetOutput()
	typeParameters := modelOutput.GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		// TODO get output layer name => for what..?
		return "", errors.New("cannot determine the name of the output layer")
	}
	return name, nil
}

// Predict ...
func (p *SemanticSegmentationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {

	if data == nil {
		return errors.New("input data nil")
	}

	gotensors, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	fst := gotensors[0]
	dims := append([]int{len(gotensors)}, fst.Shape()...)
	// debug
	// pp.Println(dims)
	// TODO: support data types other than float32
	var input []float32
	for _, t := range gotensors {
		input = append(input, t.Float32s()...)
	}

	err := p.predictor.Predict(ctx, []gotensor.Tensor{
		gotensor.New(
			gotensor.Of(gotensor.Float32),
			gotensor.WithBacking(input),
			gotensor.WithShape(dims...),
		),
	})
	if err != nil {
		return err
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *SemanticSegmentationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, errors.New("cannot get prediction output")
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.New("cannot get the labels")
	}

	outputarray := outputs[0].Data().([]float32)
	outputbatch := outputs[0].Shape()[0]
	outputfeature := outputs[0].Shape()[1]
	outputheight := outputs[0].Shape()[2]
	outputwidth := outputs[0].Shape()[3]

	// convert the output in order to make it compatible with CreateSemanticSegmentFeatures function call
	masks := make([][][]int64, outputbatch)
	for b := 0; b < outputbatch; b++ {
		masks[b] = make([][]int64, outputheight)
		for h := 0; h < outputheight; h++ {
			masks[b][h] = make([]int64, outputwidth)
			for w := 0; w < outputwidth; w++ {
				idx := 0
				cur := outputarray[b*outputfeature*outputheight*outputwidth+idx*outputheight*outputwidth+h*outputwidth+w]
				for f := 1; f < outputfeature; f++ {
					if outputarray[b*outputfeature*outputheight*outputwidth+f*outputheight*outputwidth+h*outputwidth+w] > cur {
						idx = f
						cur = outputarray[b*outputfeature*outputheight*outputwidth+idx*outputheight*outputwidth+h*outputwidth+w]
					}
				}
				masks[b][h][w] = int64(idx)
			}
		}
	}

	return p.CreateSemanticSegmentFeatures(ctx, masks, labels)
}

// ReadPredictedFeaturesAsMap ...
func (p *SemanticSegmentationPredictor) ReadPredictedFeaturesAsMap(ctx context.Context) (map[string]interface{}, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features_as_map")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}

	res := make(map[string]interface{})
	res["outputs"] = outputs
	res["labels"] = p.labels

	return res, nil
}

// Reset ...
func (p *SemanticSegmentationPredictor) Reset(ctx context.Context) error {
	return nil
}

// Close ...
func (p *SemanticSegmentationPredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

// Modality ...
func (p SemanticSegmentationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageSemanticSegmentationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := pytorch.FrameworkManifest
		agent.AddPredictor(framework, &SemanticSegmentationPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
