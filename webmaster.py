import os
import argparse
import asyncio
import logging
from dotenv import load_dotenv
import json 
from datetime import datetime
load_dotenv()
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import utils
from src.utils.agent_state import AgentState
from src.utils.default_config_settings import default_config


import base64
from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot
from google import genai

# Global variables for persisten
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def gemini_ai_chat_completions(chatPayload):

    api_key = chatPayload.get("api_key", GEMINI_API_KEY)
    model = chatPayload.get("model","gemini-2.0-flash")
    messages = chatPayload.get("messages")
    temperature = chatPayload.get("temperature",0)

    
    client = genai.Client(api_key=api_key)
    completion =client.models.generate_content(
            model=model,
            contents=messages,
            config={"temperature": temperature}
        )
    return (completion.text)
logger = logging.getLogger(__name__)

# Global variables for browser persistence
_global_browser = None
_global_browser_context = None
_global_agent = None
_global_agent_state = AgentState()

import os
import time
import tesserocr
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def extract_text_from_screenshots(screenshots_folder):
    print("Extracting text from screenshots...")
    start_time = time.time()

    # Gather all image file paths from the folder
    images = []
    for file in os.listdir(screenshots_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            images.append(os.path.join(screenshots_folder, file))
    
    if not images:
        print("No images found in the folder.")
        return ""

    # Function to perform OCR on a single image
    def ocr_image(image_path):
        try:
            with Image.open(image_path) as img:
                # Adjust the path below if your Tesseract installation is located elsewhere
                with tesserocr.PyTessBaseAPI(path=r"C:/Program Files/Tesseract-OCR/tessdata") as api:
                    api.SetImage(img)
                    return api.GetUTF8Text()
        except Exception as e:
            print(f"OCR failed for {image_path}: {str(e)}")
            return ""

    # Function to process a batch of images concurrently
    def process_images_batch(images_batch):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(ocr_image, images_batch))

    content = []
    batch_size = 5  # Process images in batches of 5
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        content.extend(process_images_batch(batch))
    
    final_content = "".join(content)
    end_time = time.time()
    print(f"OCR completed in {end_time - start_time} seconds")
    
    # Write the OCR output to a text file in the same folder
    output_file = os.path.join(screenshots_folder, "ocr_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    print(f"OCR output written to {output_file}")
    return final_content

def base64_to_image(base64_string, output_folder, output_filename):
    # If the string has a header like "data:image/png;base64,", remove it.
    if base64_string.startswith("data:"):
        header, base64_data = base64_string.split(",", 1)
    else:
        base64_data = base64_string

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_data)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create the full output path
    output_path = os.path.join(output_folder, output_filename)

    # Write the bytes to a file
    with open(output_path, "wb") as image_file:
        image_file.write(image_bytes)

    print(f"Image saved to {output_path}")



async def run_agent(config):
    """Main function to run the browser agent based on configuration"""
    global _global_browser, _global_browser_context, _global_agent, _global_agent_state

    try:
        llm = utils.get_llm_model(
            provider=config['llm_provider'],
            model_name=config['llm_model_name'],
            num_ctx=config['llm_num_ctx'],
            temperature=config['llm_temperature'],
            base_url=config['llm_base_url'],
            api_key=config['llm_api_key'],
        )

        extra_chromium_args = [f"--window-size={config['window_w']},{config['window_h']}"]
        if config['use_own_browser']:
            chrome_path = os.getenv("CHROME_PATH", None) or None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        # Initialize browser
        if _global_browser is None:
            browser_class = Browser if config['agent_type'] == "org" else CustomBrowser
            _global_browser = browser_class(
                config=BrowserConfig(
                    headless=config['headless'],
                    disable_security=config['disable_security'],
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        # Create browser context
        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=config['save_trace_path'],
                    save_recording_path=config['save_recording_path'],
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=config['window_w'],
                        height=config['window_h']
                    ),
                )
            )

        # Initialize agent
        if config['agent_type'] == "org":
            _global_agent = Agent(
                task=config['task'],
                llm=llm,
                use_vision=config['use_vision'],
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=config['max_actions_per_step'],
                tool_calling_method=config['tool_calling_method']
            )
        else:
            controller = CustomController()
            _global_agent = CustomAgent(
                task=config['task'],
                add_infos=config['add_infos'],
                use_vision=config['use_vision'],
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=config['max_actions_per_step'],
                tool_calling_method=config['tool_calling_method']
            )

        # Run the agent
        history = await _global_agent.run(max_steps=config['max_steps'])

        # Save results
        if config['save_agent_history_path']:
            history_file = os.path.join(config['save_agent_history_path'], f"{_global_agent.agent_id}.json")
            _global_agent.save_history(history_file)
            logger.info(f"Saved agent history to {history_file}")

        return {
            'final_result': history.final_result(),
            'errors': history.errors(),
            'actions': history.model_actions(),
            'thoughts': history.model_thoughts(),
            'history_file':history_file
        }

    finally:
        # Cleanup resources
        if not config['keep_browser_open']:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
            if _global_browser:
                await _global_browser.close()
                _global_browser = None

def main(main_prompt,helper_prompt,find_prompt):
    parser = argparse.ArgumentParser(description="Browser Automation Agent")
    
    # Agent configuration
    parser.add_argument('--agent-type', choices=['org', 'custom'], default='custom')
    parser.add_argument('--task', default=main_prompt, help="Task description for the agent")
    parser.add_argument('--add-infos', default=helper_prompt, help="Additional information for custom agent")
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--max-actions-per-step', type=int, default=10)
    parser.add_argument('--use-vision', action='store_true', help="Enable vision capabilities")
    parser.add_argument('--tool-calling-method', default='auto', choices=['auto', 'json_schema', 'function_calling'])

    # LLM configuration
    parser.add_argument('--llm-provider', default='google', choices=utils.model_names.keys())
    parser.add_argument('--llm-model', default="gemini-2.0-flash-exp", help="Model name to use")
    parser.add_argument('--llm-temperature', type=float, default=0)
    parser.add_argument('--llm-base-url', default="", help="Base URL for LLM API")
    parser.add_argument('--llm-api-key', default="AIzaSyD6UtTTbxSBnmViTkMkNBeagK_RA3AXsDw", help="API key for LLM provider")
    parser.add_argument('--llm-num-ctx', type=int, default=2048, help="Number of context tokens for LLM")

    # Browser configuration
    parser.add_argument('--use-own-browser', action='store_true', default=True)
    parser.add_argument('--keep-browser-open', action='store_true', default=True)
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--disable-security', action='store_true', default=True)
    parser.add_argument('--window-w', type=int, default=1080)
    parser.add_argument('--window-h', type=int, default=1080)

    # Output configuration
    parser.add_argument('--save-recording-path', default="tmp/save_recording", help="Path to save browser recordings")
    parser.add_argument('--save-trace-path', default="tmp/save_trace", help="Path to save browser traces")
    parser.add_argument('--save-history-path', default="tmp/save_history", help="Path to save agent history")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config dictionary from args
    config = {
        'agent_type': args.agent_type,
        'llm_num_ctx': args.llm_num_ctx,
        'task': args.task,
        'add_infos': args.add_infos,
        'max_steps': args.max_steps,
        'max_actions_per_step': args.max_actions_per_step,
        'use_vision': args.use_vision,
        'tool_calling_method': args.tool_calling_method,
        'llm_provider': args.llm_provider,
        'llm_model_name': args.llm_model,
        'llm_temperature': args.llm_temperature,
        'llm_base_url': args.llm_base_url,
        'llm_api_key': args.llm_api_key,
        'use_own_browser': args.use_own_browser,
        'keep_browser_open': args.keep_browser_open,
        'headless': args.headless,
        'disable_security': args.disable_security,
        'window_w': args.window_w,
        'window_h': args.window_h,
        'save_recording_path': args.save_recording_path,
        'save_trace_path': args.save_trace_path,
        'save_agent_history_path': args.save_history_path
    }

    # Run the agent
    try:
        result = asyncio.run(run_agent(config))
        print("\nResults:")
        print(f"Final Result: {result['final_result']}")
        print(f"Actions Taken: {result['actions']}")
        print(f"Thought Process: {result['thoughts']}")
        if result['errors']:
            print(f"\nErrors encountered:\n{result['errors']}")
        history_path=result['history_file']
        with open(history_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
        
        # output_folder = "E:/Rishit/Legitt/code/Operator/web-ui/screenshots"
        # new_folder_name = f"run_at_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # new_folder_path = os.path.join(output_folder, new_folder_name)
        # os.makedirs(new_folder_path, exist_ok=True)
        # for i in range(len(json_data["history"])):
        #     screenshot_base64 = json_data["history"][i]["state"]["screenshot"]
        #     output_filename = f"image_{i+1}.png"
            
        #     base64_to_image(screenshot_base64,new_folder_path,output_filename)
        # ocr_result=extract_text_from_screenshots(new_folder_path)

        # for item in json_data.get("history", []):
        #     if "state" in item and "screenshot" in item["state"]:
        #         del item["state"]["screenshot"]
        chatPayload={"messages":f"""You have been provided a detail from Webpages, You task is to Find the relevant information asked by the user,from the Webpage data
                     
                     Webpage data:{result['actions']}{result['final_result']}{result['thoughts']}")

                    What user wants to Find:{find_prompt}"""}
        answer= gemini_ai_chat_completions(chatPayload)
        print(answer)
    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}")
        raise
if __name__ == '__main__':
    main_prompt="""Access Legitt AI Website

If logged out, log in using the following credentials:
Email ID: rishit.dass@legittai.com
Password:
2️ Retrieve and Compile Data (Step-by-Step):

User Profile Details: Gather all generic information related to the user.
Document List: Extract the full names of every document associated with the user ID.
User Reports: Locate and compile all reports linked to the user ID.
3️ Final Output:

Provide a comprehensive, well-structured, and 100% accurate report summarizing all findings.
Ensure deep research and verification of all extracted data.
Maintain clarity, accuracy, and completeness in the final report.
"""
    helper_prompt=""
    find_prompt="""    1. Find the Generic Details about the User
    2. Find all the documents present on user's Id
    3. Find the reports of user id"""
    main(main_prompt,helper_prompt,find_prompt)