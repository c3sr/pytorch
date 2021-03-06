package predictor

// import (
// 	"context"
// 	"os"
// 	"path/filepath"
// 	"testing"

// 	"github.com/c3sr/dlframework/framework/options"
// 	"github.com/c3sr/image"
// 	"github.com/c3sr/image/types"
// 	nvidiasmi "github.com/c3sr/nvidia-smi"
// 	py "github.com/c3sr/pytorch"
// 	"github.com/stretchr/testify/assert"
// 	gotensor "gorgonia.org/tensor"
// )

// func TestInstanceSegmentation(t *testing.T) {
// 	py.Register()
// 	model, err := py.FrameworkManifest.FindModel("mask_rcnn_inception_v2_coco:1.0")
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, model)

// 	device := options.CPU_DEVICE
// 	if nvidiasmi.HasGPU {
// 		device = options.CUDA_DEVICE
// 	}

// 	batchSize := 1
// 	ctx := context.Background()
// 	opts := options.New(options.Context(ctx),
// 		options.Device(device, 0),
// 		options.BatchSize(batchSize))

// 	predictor, err := NewInstanceSegmentationPredictor(*model, options.WithOptions(opts))
// 	assert.NoError(t, err)
// 	assert.NotEmpty(t, predictor)
// 	defer predictor.Close()

// 	imgDir, _ := filepath.Abs("./_fixtures")
// 	imgPath := filepath.Join(imgDir, "lane_control.jpg")
// 	r, err := os.Open(imgPath)
// 	if err != nil {
// 		panic(err)
// 	}
// 	img, err := image.Read(r)
// 	if err != nil {
// 		panic(err)
// 	}

// 	height := img.Bounds().Dy()
// 	width := img.Bounds().Dx()
// 	channels := 3
// 	input := make([]gotensor.Tensor, batchSize)
// 	imgBytes := img.(*types.RGBImage).Pix

// 	for ii := 0; ii < batchSize; ii++ {
// 		input[ii] = gotensor.New(
// 			gotensor.WithShape(height, width, channels),
// 			gotensor.WithBacking(imgBytes),
// 		)
// 	}

//  joined, err := gotensor.Concat(0, input[0], input[1:]...)
//  if err != nil {
//    return
//  }
//  joined.Reshape(append([]int{len(input)}, input[0].Shape()...)...)

//  err = predictor.Predict(ctx, []gotensor.Tensor{joined})

// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	pred, err := predictor.ReadPredictedFeatures(ctx)
// 	assert.NoError(t, err)
// 	if err != nil {
// 		return
// 	}

// 	assert.InDelta(t, float32(0.998607), pred[0][0].GetProbability(), 0.001)
// }
