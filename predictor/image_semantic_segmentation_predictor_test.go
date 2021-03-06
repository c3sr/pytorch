package predictor

import (
	"context"
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

func TestSemanticSegmentation(t *testing.T) {
	py.Register()
	model, err := py.FrameworkManifest.FindModel("TorchVision_DeepLabv3_Resnet101:1.0")
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

	predictor, err := NewSemanticSegmentationPredictor(*model, options.WithOptions(opts))
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

	height := img.Bounds().Dy()
	width := img.Bounds().Dx()
	channels := 3

	input := make([]gotensor.Tensor, batchSize)
	imgFloats, err := normalizeImageCHW(img, preprocessOpts.MeanImage, preprocessOpts.Scale)
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

	sseg := pred[0][0].GetSemanticSegment()
	intMask := sseg.GetIntMask()

	assert.Equal(t, int32(7), intMask[247039])
}
