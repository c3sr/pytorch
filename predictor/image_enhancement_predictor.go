package predictor

import (
	"context"
	"io"
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

// ImageEnhancementPredictor ...
type ImageEnhancementPredictor struct {
	common.ImagePredictor
	predictor *gopytorch.Predictor
	images    interface{}
}

// NewImageEnhancementPredictor ...
func NewImageEnhancementPredictor(model dlframework.ModelManifest, os ...options.Option) (common.Predictor, error) {
	opts := options.New(os...)
	ctx := opts.Context()

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

	predictor := new(ImageEnhancementPredictor)

	return predictor.Load(ctx, model, os...)
}

// Download ...
func (p *ImageEnhancementPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ImageEnhancementPredictor{
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
func (p *ImageEnhancementPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImageEnhancementPredictor{
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

func (p *ImageEnhancementPredictor) download(ctx context.Context) error {
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

	return nil
}

func (p *ImageEnhancementPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "load predictor"),
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
func (p ImageEnhancementPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
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
func (p ImageEnhancementPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
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

func makeUniformImage() [][][][]float32 {
	images := make([][][][]float32, 10)
	width := 1000
	height := 1000
	for ii := range images {
		sl := make([][][]float32, height)
		for jj := range sl {
			el := make([][]float32, width)
			for kk := range el {
				el[kk] = []float32{1, 0, 1}
			}
			sl[jj] = el
		}
		images[ii] = sl
	}
	return images
}

// Predict ...
func (p *ImageEnhancementPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	if data == nil {
		return errors.New("input data nil")
	}

	gotensors, ok := data.([]gotensor.Tensor)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	return p.predictor.Predict(ctx, gotensors)
}

// ReadPredictedFeatures ...
func (p *ImageEnhancementPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}

	outputarray := outputs[0].Data().([]float32)
	outputbatch := outputs[0].Shape()[0]
	outputchannels := outputs[0].Shape()[1]
	outputheight := outputs[0].Shape()[2]
	outputwidth := outputs[0].Shape()[3]

	// convert 1D array to a 4D array in order to make it compatible with CreateRawImageFeatures function call
	e := make([][][][]float32, outputbatch)
	for b := 0; b < outputbatch; b++ {
		e[b] = make([][][]float32, outputheight)
		for h := 0; h < outputheight; h++ {
			e[b][h] = make([][]float32, outputwidth)
			for w := 0; w < outputwidth; w++ {
				e[b][h][w] = make([]float32, 3)
			}
		}
	}
	for b := 0; b < outputbatch; b++ {
		for h := 0; h < outputheight; h++ {
			for w := 0; w < outputwidth; w++ {
				e[b][h][w][0] = outputarray[b*outputheight*outputwidth*outputchannels+0*outputheight*outputwidth+h*outputwidth+w]
				e[b][h][w][1] = outputarray[b*outputheight*outputwidth*outputchannels+1*outputheight*outputwidth+h*outputwidth+w]
				e[b][h][w][2] = outputarray[b*outputheight*outputwidth*outputchannels+2*outputheight*outputwidth+h*outputwidth+w]
			}
		}
	}

	return p.CreateRawImageFeatures(ctx, e)
}

// ReadPredictedFeaturesAsMap ...
func (p *ImageEnhancementPredictor) ReadPredictedFeaturesAsMap(ctx context.Context) (map[string]interface{}, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features_as_map")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}

	res := make(map[string]interface{})
	res["outputs"] = outputs

	return res, nil
}

// Reset ...
func (p *ImageEnhancementPredictor) Reset(ctx context.Context) error {
	return nil
}

// Close ...
func (p *ImageEnhancementPredictor) Close() error {
	return nil
}

// Modality ...
func (p ImageEnhancementPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageEnhancementModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := pytorch.FrameworkManifest
		agent.AddPredictor(framework, &ImageEnhancementPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
