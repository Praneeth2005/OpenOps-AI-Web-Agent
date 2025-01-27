import pdb
import logging

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt
from src.browser.config import BrowserPersistenceConfig
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import (
    default_config,
    load_config_from_file,
    save_config_to_file,
    save_current_config,
    update_ui_from_config,
)
from src.utils.utils import (
    update_model_dropdown,
    get_latest_files,
    capture_screenshot,
)

from dotenv import load_dotenv
load_dotenv()

# Global variables for persistence
_global_browser = None
_global_browser_context = None

# Create the global agent state instance
_global_agent_state = AgentState()

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested – the agent will halt at the next safe point."
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_call_in_content
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_call_in_content=tool_call_in_content
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_call_in_content=tool_call_in_content,
                model_name=llm_model_name  # Pass model_name to run_custom_agent
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            "",                                         # final_result
            errors,                                     # errors
            "",                                         # model_actions
            "",                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # trace_file
            None,                                       # history_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def run_org_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_call_in_content
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state
        
        # Clear any previous stop request
        _global_agent_state.clear_stop()

        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=[f"--window-size={window_w},{window_h}"],
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser=_global_browser,
            browser_context=_global_browser_context,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get(".zip"), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return "", errors, "", "", None, None
    finally:
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_call_in_content,
    model_name=None  # Add model_name parameter
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state

        # Clear any previous stop request
        _global_agent_state.clear_stop()

        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=[f"--window-size={window_w},{window_h}"],
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        # Create and run agent
        agent = CustomAgent(
            task=task,
            add_infos=add_infos,
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content,
            agent_state=_global_agent_state,
            model_name=model_name  # Pass model_name to CustomAgent
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get(".zip"), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return "", errors, "", "", None, None
    finally:
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_call_in_content
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        # If we're NOT headless, we simply run the agent and update the HTML preview
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_call_in_content=tool_call_in_content
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        # If we are headless, attempt to stream screenshots
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_call_in_content=tool_call_in_content
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None

            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stop", interactive=True),  # Re-enable stop button
                        gr.update(interactive=True)  # Re-enable run button
                    ]
                await asyncio.sleep(0.05)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                (
                    final_result,
                    errors,
                    model_actions,
                    model_thoughts,
                    latest_videos,
                    trace,
                    history_file,
                    stop_button,
                    run_button,
                ) = result
            except Exception as e:
                errors = f"Agent error: {str(e)}"
                stop_button = gr.update(value="Stop", interactive=True)
                run_button = gr.update(interactive=True)

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                stop_button,
                run_button,
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True),  # Re-enable run button
            ]

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base(),
}

async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None

def create_ui(config, theme_name="Ocean"):
    """
    UI with friendlier, more descriptive labels and headings.
    """
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
        background: linear-gradient(to bottom right, #D9AFD9, #97D9E1);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        color: #333;
        font-family: "Open Sans", sans-serif;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    /* Accordion titles */
    .gr-accordion .label-wrap {
        background-color: rgba(255,255,255,0.8);
        border-radius: 6px;
        margin-bottom: 8px;
        cursor: pointer;
    }
    .gr-accordion .label-wrap:hover {
        background-color: rgba(255,255,255,1);
    }
    /* Slider accent */
    input[type=range] {
        accent-color: #6FA1F2;
    }
    /* Buttons */
    .gr-button {
        background: #6FA1F2 !important;
        color: #fff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    .gr-button:hover {
        background: #4E87EE !important;
    }
    /* Textboxes, dropdowns, etc. */
    .gr-textbox textarea, .gr-dropdown select, .gr-number input, .gr-file input {
        background-color: rgba(255,255,255,0.9) !important;
        border-radius: 5px !important;
        border: 1px solid #ccc !important;
    }
    .gr-textbox label, .gr-slider label, .gr-checkbox label, .gr-dropdown label {
        font-weight: 600;
        color: #333;
    }
    /* Gallery items */
    .gr-gallery img {
        border-radius: 8px;
    }
    /* Video preview */
    .gr-video video {
        border-radius: 8px;
    }
    """

    js = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    with gr.Blocks(theme=theme_map[theme_name], css=css, js=js) as demo:
        # Header
        gr.Markdown(
            """
            # 🌐 OpenOps Agent 
            ### Seamlessly orchestrate your AI‐powered tasks—empowering you to operate, automate, and innovate.
            """,
            elem_classes=["header-text"]
        )

        # Agent Settings
        with gr.Accordion("1. AI Agent Configuration", open=False):
            agent_type = gr.Radio(
                ["org", "custom"],
                label="Select Agent Type",
                value=config["agent_type"],
                info="Choose either the default agent or a custom setup.",
            )
            max_steps = gr.Slider(
                minimum=1,
                maximum=200,
                value=config["max_steps"],
                step=1,
                label="Maximum Steps",
                info="How many steps the AI can take before stopping.",
            )
            max_actions_per_step = gr.Slider(
                minimum=1,
                maximum=20,
                value=config["max_actions_per_step"],
                step=1,
                label="Actions per Step",
                info="Maximum number of browser actions each step can contain.",
            )
            use_vision = gr.Checkbox(
                label="Enable Visual Processing",
                value=config["use_vision"],
                info="Allow the AI to analyze and process web visuals.",
            )
            tool_call_in_content = gr.Checkbox(
                label="Tool Calls in AI Output",
                value=config["tool_call_in_content"],
                info="Let the AI embed tool calls within its output text.",
            )

        # LLM Configuration
        with gr.Accordion("2. Language Model Settings", open=False):
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in utils.model_names.items()],
                label="Language Model Provider",
                value=config["llm_provider"],
                info="Pick the provider that delivers your LLM.",
            )
            llm_model_name = gr.Dropdown(
                label="Model Name",
                choices=utils.model_names["openai"],
                value=config["llm_model_name"],
                interactive=True,
                allow_custom_value=True,  # Allow custom model names
                info="Choose or type a custom LLM model name.",
            )
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=config["llm_temperature"],
                step=0.1,
                label="Creativity (Temperature)",
                info="Adjust output randomness: lower is more deterministic, higher is more creative.",
            )
            with gr.Row():
                llm_base_url = gr.Textbox(
                    label="Custom API URL",
                    value=config["llm_base_url"],
                    info="Endpoint for the language model, if not using defaults.",
                )
                llm_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    value=config["llm_api_key"],
                    info="Provide your model API key (or leave blank to rely on .env).",
                )

        # Browser Settings
        with gr.Accordion("3. Browser Environment", open=False):
            with gr.Row():
                use_own_browser = gr.Checkbox(
                    label="Use Existing Browser Instance",
                    value=config["use_own_browser"],
                    info="Allow the AI to connect to your existing Chromium browser.",
                )
                keep_browser_open = gr.Checkbox(
                    label="Keep Browser Open",
                    value=config["keep_browser_open"],
                    info="Keep the browser session open after tasks complete.",
                )
                headless = gr.Checkbox(
                    label="Headless Mode",
                    value=config["headless"],
                    info="Run the browser in the background without a visible window.",
                )
                disable_security = gr.Checkbox(
                    label="Disable Security Features",
                    value=config["disable_security"],
                    info="Disable certain security flags (less secure).",
                )
                enable_recording = gr.Checkbox(
                    label="Enable Video Recording",
                    value=config["enable_recording"],
                    info="Record the browser session as a video file.",
                )
            with gr.Row():
                window_w = gr.Number(
                    label="Browser Window Width",
                    value=config["window_w"],
                    info="Set the window width in pixels.",
                )
                window_h = gr.Number(
                    label="Browser Window Height",
                    value=config["window_h"],
                    info="Set the window height in pixels.",
                )
            save_recording_path = gr.Textbox(
                label="Save Recordings to Folder",
                placeholder="e.g. ./tmp/record_videos",
                value=config["save_recording_path"],
                info="Specify where the recorded browser sessions should be stored.",
                interactive=True,
            )
            save_trace_path = gr.Textbox(
                label="Save Trace Data to Folder",
                placeholder="e.g. ./tmp/traces",
                value=config["save_trace_path"],
                info="Folder for storing performance or trace logs.",
                interactive=True,
            )
            save_agent_history_path = gr.Textbox(
                label="Save Agent History to Folder",
                placeholder="e.g., ./tmp/agent_history",
                value=config["save_agent_history_path"],
                info="Where the agent’s step‐by‐step history is stored.",
                interactive=True,
            )

        # Configuration Management (Load/Save)
        with gr.Accordion("4. Project Configuration", open=False):
            config_file_input = gr.File(
                label="Load Settings from File",
                file_types=[".pkl"],
                interactive=True,
            )
            load_config_button = gr.Button("Import Existing Settings", variant="primary")
            save_config_button = gr.Button("Save Current Settings", variant="primary")
            config_status = gr.Textbox(
                label="Config Status",
                lines=2,
                interactive=False,
            )

        # Agent Task Section
        gr.Markdown("## Run the AI Agent")
        with gr.Row():
            task = gr.Textbox(
                label="Task Instructions",
                lines=4,
                placeholder="Describe the objective...",
                value=config["task"],
                info="Explain the specific task(s) you want the AI agent to accomplish.",
            )
            add_infos = gr.Textbox(
                label="Additional Hints",
                lines=3,
                placeholder="Optional extra context for the AI...",
                info="Provide more details or constraints to guide the AI’s approach.",
            )

        with gr.Row():
            run_button = gr.Button("▶️ Start Agent", variant="primary", scale=2)
            stop_button = gr.Button("⏹️ Stop Execution", variant="stop", scale=1)

        # Browser Live View
        browser_view = gr.HTML(
            value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
            label="Live Browser Feed",
        )

        # Results Section
        gr.Markdown("## Output & Logs")
        with gr.Row():
            with gr.Column():
                final_result_output = gr.Textbox(
                    label="Results Summary", lines=3, show_label=True
                )
            with gr.Column():
                errors_output = gr.Textbox(
                    label="Error Log", lines=3, show_label=True
                )
        with gr.Row():
            with gr.Column():
                model_actions_output = gr.Textbox(
                    label="Recorded Actions", lines=3, show_label=True
                )
            with gr.Column():
                model_thoughts_output = gr.Textbox(
                    label="Agent’s Reasoning", lines=3, show_label=True
                )
        with gr.Row():
            recording_display = gr.Video(label="Most Recent Recording")
            trace_file = gr.File(label="Debug Trace File")
            agent_history_file = gr.File(label="Agent Run History")

        # Recordings Section
        gr.Markdown("## Recorded Sessions")
        def list_recordings(save_recording_path):
            if not os.path.exists(save_recording_path):
                return []
            # Get all video files
            recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            # Sort recordings by creation time (oldest first)
            recordings.sort(key=os.path.getctime)
            # Number them
            numbered_recordings = []
            for idx, recording in enumerate(recordings, start=1):
                filename = os.path.basename(recording)
                numbered_recordings.append((recording, f"{idx}. {filename}"))
            return numbered_recordings

        recordings_gallery = gr.Gallery(
            label="Session Recordings",
            value=list_recordings(config["save_recording_path"]),
            columns=3,
            height="auto",
            object_fit="contain",
        )
        refresh_button = gr.Button("🔄 Refresh List", variant="secondary")

        # Callbacks
        refresh_button.click(
            fn=list_recordings,
            inputs=save_recording_path,
            outputs=recordings_gallery
        )

        # Attach the callback to the LLM provider dropdown
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
            inputs=[llm_provider, llm_api_key, llm_base_url],
            outputs=llm_model_name
        )

        # Enable/disable the Recording Path textbox based on the "Enable Recording" checkbox
        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            inputs=enable_recording,
            outputs=save_recording_path
        )

        # If the user changes the 'use_own_browser' or 'keep_browser_open', close the global browser:
        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

        # Stop button click
        stop_button.click(
            fn=stop_agent,
            inputs=[],
            outputs=[errors_output, stop_button, run_button],
        )

        # Run button click
        run_button.click(
            fn=run_with_stream,
            inputs=[
                agent_type, llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                save_recording_path, save_agent_history_path, save_trace_path,
                enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_call_in_content
            ],
            outputs=[
                browser_view,           # Browser view
                final_result_output,    # Final result
                errors_output,          # Errors
                model_actions_output,   # Model actions
                model_thoughts_output,  # Model thoughts
                recording_display,      # Latest recording
                trace_file,             # Trace file
                agent_history_file,     # Agent history file
                stop_button,            # Stop button
                run_button,             # Run button
            ],
        )

        # Load config button
        load_config_button.click(
            fn=update_ui_from_config,
            inputs=[config_file_input],
            outputs=[
                agent_type,
                max_steps,
                max_actions_per_step,
                use_vision,
                tool_call_in_content,
                llm_provider,
                llm_model_name,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                enable_recording,
                window_w,
                window_h,
                save_recording_path,
                save_trace_path,
                save_agent_history_path,
                task,
                config_status,
            ],
        )

        # Save config button
        save_config_button.click(
            fn=save_current_config,
            inputs=[
                agent_type,
                max_steps,
                max_actions_per_step,
                use_vision,
                tool_call_in_content,
                llm_provider,
                llm_model_name,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                enable_recording,
                window_w,
                window_h,
                save_recording_path,
                save_trace_path,
                save_agent_history_path,
                task,
            ],
            outputs=[config_status],
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent (OpenOps Edition)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument(
        "--theme",
        type=str,
        default="Ocean",
        choices=theme_map.keys(),
        help="Theme to use for the UI",
    )
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()

    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == "__main__":
    main()
