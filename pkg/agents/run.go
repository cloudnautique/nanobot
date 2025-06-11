package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/obot-platform/nanobot/pkg/confirm"
	"github.com/obot-platform/nanobot/pkg/mcp"
	"github.com/obot-platform/nanobot/pkg/schema"
	"github.com/obot-platform/nanobot/pkg/tools"
	"github.com/obot-platform/nanobot/pkg/types"
)

type Agents struct {
	config        types.Config
	completer     types.Completer
	registry      *tools.Registry
	confirmations *confirm.Service
}

type ToolListOptions struct {
	ToolName string
	Names    []string
}

func New(completer types.Completer, registry *tools.Registry, confirmations *confirm.Service, config types.Config) *Agents {
	return &Agents{
		config:        config,
		completer:     completer,
		registry:      registry,
		confirmations: confirmations,
	}
}

func (a *Agents) addTools(ctx context.Context, req *types.CompletionRequest, agent *types.Agent) (types.ToolMappings, error) {
	toolMappings, err := a.registry.BuildToolMappings(ctx, agent.Tools)
	if err != nil {
		return nil, fmt.Errorf("failed to build tool mappings: %w", err)
	}

	for _, key := range slices.Sorted(maps.Keys(toolMappings)) {
		toolMapping := toolMappings[key]

		tool := toolMapping.Target.(mcp.Tool)
		req.Tools = append(req.Tools, types.ToolUseDefinition{
			Name:        key,
			Parameters:  schema.ValidateAndFixToolSchema(tool.InputSchema),
			Description: tool.Description,
			Attributes:  agent.ToolExtensions[key],
		})
	}

	return toolMappings, nil
}

func (a *Agents) populateRequest(ctx context.Context, run *run, previousRun *run) (types.CompletionRequest, types.ToolMappings, error) {
	req := run.Request

	if previousRun != nil {
		input := previousRun.PopulatedRequest.Input

		for _, output := range previousRun.Response.Output {
			input = append(input, output.ToInput())
		}

		for _, callID := range slices.Sorted(maps.Keys(previousRun.ToolOutputs)) {
			toolCall := previousRun.ToolOutputs[callID]
			if toolCall.Done {
				input = append(input, toolCall.Output...)
			}
		}

		input = append(input, req.Input...)
		req.Input = input
	}

	agent, ok := a.config.Agents[req.Model]
	if !ok {
		return req, nil, nil
	}

	if req.SystemPrompt != "" {
		var agentInstructions types.DynamicInstructions
		if err := json.Unmarshal([]byte(strings.TrimSpace(req.SystemPrompt)), &agentInstructions); err == nil &&
			agentInstructions.IsPrompt() {
			req.SystemPrompt = ""
			agent.Instructions = agentInstructions
		}
	}

	if req.SystemPrompt == "" && agent.Instructions.IsSet() {
		var err error
		req.SystemPrompt, err = a.registry.GetDynamicInstruction(ctx, agent.Instructions)
		if err != nil {
			return req, nil, err
		}
	}

	if req.TopP == nil && agent.TopP != nil {
		req.TopP = agent.TopP
	}

	if req.Temperature == nil && agent.Temperature != nil {
		req.Temperature = agent.Temperature
	}

	if req.Truncation == "" && agent.Truncation != "" {
		req.Truncation = agent.Truncation
	}

	if req.MaxTokens == 0 && agent.MaxTokens != 0 {
		req.MaxTokens = agent.MaxTokens
	}

	if req.ToolChoice == "" && agent.ToolChoice != "" {
		req.ToolChoice = agent.ToolChoice
	}

	if previousRun != nil {
		// Don't allow tool choice if this is a follow-on request
		req.ToolChoice = ""
	}

	if req.OutputSchema == nil && agent.Output != nil && len(agent.Output.Schema) > 0 {
		name := agent.Output.Name
		if name == "" {
			name = "output_schema"
		}
		req.OutputSchema = &types.OutputSchema{
			Name:        name,
			Description: agent.Output.Description,
			Schema:      agent.Output.Schema,
			Strict:      agent.Output.Strict,
		}
	}

	req.Model = agent.Model

	toolMapping, err := a.addTools(ctx, &req, &agent)
	if err != nil {
		return req, nil, fmt.Errorf("failed to add tools: %w", err)
	}

	// Validate and fix tool input schemas
	for i, tool := range req.Tools {
		fixedSchema := schema.ValidateAndFixToolSchema(tool.Parameters)
		req.Tools[i].Parameters = fixedSchema
	}

	return req, toolMapping, nil
}

const previousRunKey = "previous_run"

func (a *Agents) Complete(ctx context.Context, req types.CompletionRequest, opts ...types.CompletionOptions) (*types.CompletionResponse, error) {
	var (
		previousRunKey = previousRunKey + "/" + req.Model
		session        = mcp.SessionFromContext(ctx)
		stateful       = session != nil
		previousRun    *run
		currentRun     = &run{
			Request: req,
		}
	)

	if stateful && a.config.Agents[req.Model].ChatHistory != nil && !*a.config.Agents[req.Model].ChatHistory {
		stateful = false
	}

	if stateful {
		previousRun, _ = session.Get(previousRunKey).(*run)
	}

	for {
		if err := a.run(ctx, currentRun, previousRun, opts); err != nil {
			return nil, err
		}

		if err := a.toolCalls(ctx, currentRun, opts); err != nil {
			return nil, err
		}

		if currentRun.Done {
			if stateful {
				session.Set(previousRunKey, currentRun)
			}
			return currentRun.Response, nil
		}

		previousRun = currentRun
		currentRun = &run{
			Request: types.CompletionRequest{
				Model: currentRun.Request.Model,
			},
		}
	}
}

func (a *Agents) run(ctx context.Context, run *run, prev *run, opts []types.CompletionOptions) error {
	completionRequest, toolMapping, err := a.populateRequest(ctx, run, prev)
	if err != nil {
		return err
	}

	// Save the populated request to the run status
	run.PopulatedRequest = &completionRequest
	run.ToolToMCPServer = toolMapping

	resp, err := a.completer.Complete(ctx, completionRequest, opts...)
	if err != nil {
		return err
	}

	run.Response = resp
	return nil
}
