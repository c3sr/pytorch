package main

import (
	"fmt"
	"os"
	
	"github.com/rai-project/config"
	cmd "github.com/c3sr/dlframework/framework/cmd/server"
	"github.com/rai-project/logger"
	"github.com/c3sr/pytorch"
	_ "github.com/c3sr/pytorch/predictor"
	"github.com/rai-project/tracer"
	"github.com/sirupsen/logrus"
)

var (
	modelName string
	modelVersion string
	hostName, _ = os.Hostname()
	framework = pytorch.FrameworkManifest
	log *logrus.Entry
)

func main() {
	rootCmd, err := cmd.NewRootCommand(pytorch.Register, framework)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	defer tracer.Close()
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "pytorch-agent")
	})
}
