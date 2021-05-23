package predictor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/c3sr/dlframework/framework/options"
	raiimage "github.com/c3sr/image"
	"github.com/c3sr/image/types"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	py "github.com/c3sr/pytorch"
	"github.com/stretchr/testify/assert"
	gotensor "gorgonia.org/tensor"
)

func TestObjectDetection(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("MobileNet_SSD_v1.0:1.0")
	assert.NoError(t, err)
	assert.NotEmpty(t, model)

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	batchSize := 1
	ctx := context.Background()
	opts := options.New(options.Context(ctx),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := NewObjectDetectionPredictor(*model, options.WithOptions(opts))
	assert.NoError(t, err)
	assert.NotEmpty(t, predictor)
	defer predictor.Close()

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, "lane_control.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	preprocessOpts, err := predictor.GetPreprocessOptions()
	assert.NoError(t, err)
	channels := preprocessOpts.Dims[0]
	height := preprocessOpts.Dims[1]
	width := preprocessOpts.Dims[2]
	mode := preprocessOpts.ColorMode

	var imgOpts []raiimage.Option
	if mode == types.RGBMode {
		imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	} else {
		imgOpts = append(imgOpts, raiimage.Mode(types.BGRMode))
	}

	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)
	if err != nil {
		panic(err)
	}

	input := make([]gotensor.Tensor, batchSize)
	imgFloats, err := normalizeImageCHW(resized, preprocessOpts.MeanImage, preprocessOpts.Scale)
	if err != nil {
		panic(err)
	}

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.WithShape(channels, height, width),
			gotensor.WithBacking(imgFloats),
		)
	}

	joined, err := gotensor.Concat(0, input[0], input[1:]...)
	if err != nil {
		return
	}
	joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

	err = predictor.Predict(ctx, []gotensor.Tensor{joined})

	assert.NoError(t, err)
	if err != nil {
		return
	}

	pred, err := predictor.ReadPredictedFeatures(ctx)
	assert.NoError(t, err)
	if err != nil {
		return
	}

	for ii, cnt := 0, 0; ii < len(pred[0]) && cnt < 3; ii++ {
		if pred[0][ii].GetProbability() >= 0.5 {
			cnt++
			fmt.Printf("|                             | ./_fixtures/lane_control.jpg           | %s   | %.3f | %.3f | %.3f | %.3f | %.3f       |\n",
				pred[0][ii].GetBoundingBox().GetLabel(),
				pred[0][ii].GetBoundingBox().GetXmin(),
				pred[0][ii].GetBoundingBox().GetXmax(),
				pred[0][ii].GetBoundingBox().GetYmin(),
				pred[0][ii].GetBoundingBox().GetYmax(),
				pred[0][ii].GetProbability())
		}
	}
}
