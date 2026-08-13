// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"strconv"

	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/logging"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/server"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
)

func main() {
	// Initialize slog with ContextHandler
	jsonHandler := slog.NewJSONHandler(os.Stdout, nil)
	ctxHandler := logging.NewContextHandler(jsonHandler)
	slog.SetDefault(slog.New(ctxHandler))

	port := flag.Int("port", 9001, "The port to listen on")
	deploymentMode := flag.String("deployment-mode", "standalone", "Deployment mode ('standalone' or 'k8s')")
	defaultBackend := flag.String("default-backend", string(backends.BackendCuda),
		"Backend used when a request carries no backend_config (the orchestrator never sends one, "+
			"so this selects the backend for orchestrator-driven snapshots/restores)")
	flag.Parse()

	depMode := *deploymentMode
	if envDepMode := os.Getenv("DEPLOYMENT_MODE"); envDepMode != "" {
		depMode = envDepMode
	}

	// DEFAULT_BACKEND overrides the flag, mirroring DEPLOYMENT_MODE: the Helm
	// chart configures the agent through env vars, not flags.
	defBackend := backends.BackendType(*defaultBackend)
	if envBackend := os.Getenv("DEFAULT_BACKEND"); envBackend != "" {
		defBackend = backends.BackendType(envBackend)
	}

	// AGENT_PORT overrides the flag, mirroring DEPLOYMENT_MODE: the Helm
	// chart configures the agent through env vars, not flags.
	listenPort := *port
	if envPort := os.Getenv("AGENT_PORT"); envPort != "" {
		p, err := strconv.Atoi(envPort)
		if err != nil {
			slog.Error("Invalid AGENT_PORT", "value", envPort, "error", err)
			os.Exit(1)
		}
		listenPort = p
	}

	if depMode != "standalone" && depMode != "k8s" {
		slog.Error("Invalid deployment mode, must be 'standalone' or 'k8s'", "mode", depMode)
		os.Exit(1)
	}
	ctx := context.Background()

	// The channel registry is shared between the app-channel backend and the
	// server's WorkloadChannel RPC handler.
	channelRegistry := backends.NewChannelRegistry()
	registeredBackends := map[backends.BackendType]backends.Backend{
		backends.BackendCuda:        backends.NewCudaCheckpoint(),
		backends.BackendNoop:        backends.NewNoopBackend(),
		backends.BackendAppEndpoint: backends.NewAppEndpointBackend(),
		backends.BackendAppChannel:  backends.NewAppChannelBackend(channelRegistry),
		backends.BackendTpu:         backends.NewTpuCheckpoint(),
	}
	if _, ok := registeredBackends[defBackend]; !ok {
		slog.Error("Invalid default backend", "backend", defBackend)
		os.Exit(1)
	}

	// On TPU nodes there is no NVML: swap in TPU process discovery (libtpu
	// control threads + /dev/vfio fds) for the watcher and PID resolution.
	if accel := os.Getenv("ACCELERATOR_TYPE"); accel == "tpu" {
		utils.GetPodPIDs = utils.GetPodTpuPIDs
		utils.HasGPUProcesses = utils.HasTpuProcesses
		slog.InfoContext(ctx, "Using TPU process discovery", "acceleratorType", accel)
	}

	slog.InfoContext(ctx, "Starting Snapshot Agent", "port", listenPort, "deploymentMode", depMode, "defaultBackend", defBackend)
	if err := server.StartServer(ctx, listenPort, registeredBackends, defBackend, depMode, channelRegistry); err != nil {
		slog.ErrorContext(ctx, "Failed to start server", "error", err)
		os.Exit(1)
	}
}
