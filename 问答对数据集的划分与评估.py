#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Midscene AI Test Code Generator
Generate high-quality midscene + Playwright test code based on natural language descriptions
Support bidirectional conversion: Code ↔ Natural Language
Support direct execution of generated code
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import tempfile
import os
import json
import threading
import time
# Removed langchain dependency
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

# 使用相对路径和动态获取当前工作目录
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR  # 当前脚本所在目录作为项目目录

# Model configuration - removed unused variables

class MidsceneCodeGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Midscene AI Test Code Generator")
        self.root.geometry("1920x1080")
        
        # Configuration variables
        self.config = {
            "api_base": tk.StringVar(value="https://api.siliconflow.cn/v1"),
            "openai_model": tk.StringVar(value="Pro/Qwen/Qwen2.5-VL-7B-Instruct"),  # OpenAI model
            "openai_api_key": tk.StringVar(value="sk-ltoztgvvounfndpltgjhpgiargiddnozallzwdsaozzocxqz"),  # OpenAI API key - user should input their own key
            "base_url": tk.StringVar(value="https://example.com"),
            "test_name": tk.StringVar(value="example-test"),
            # Recording related configuration
            "enable_recording": tk.BooleanVar(value=True),
            "recording_output_dir": tk.StringVar(value="recordings")
        }
        
        # Execution related variables
        self.execution_process = None
        self.temp_test_dir = None
        self.saved_file_path = None
        self.report_dir = None  # Report directory for test results
        # Recording related variables
        self.recording_process = None
        self.recording_file_path = None
        
        # Set theme
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        
        # Create LLM chains
        self.code_to_text_chain = self._create_code_to_text_chain()
        self.text_to_code_chain = self._create_text_to_code_chain()
        
        self._setup_ui()
        
        # Check initial environment status
        self._update_environment_status()
        
    def _create_code_to_text_chain(self):
        """Create LLM for code to natural language conversion"""
        api_key = self.config["openai_api_key"].get()
        if not api_key:
            raise ValueError("OpenAI API key is required. Please configure it in the settings.")
        
        llm = OpenAILike(
            model=self.config["openai_model"].get(),
            api_base=self.config["api_base"].get(),
            api_key=api_key,
            timeout=120,
            is_chat_model=True,  # 设置为True以支持聊天接口
            is_function_calling_model=False,  # 根据需要设置
            context_window=128000  # 根据模型设置合适的上下文窗口
        )
        
        return llm
        
    def _create_text_to_code_chain(self):
        """Create LLM for natural language to midscene code conversion"""
        api_key = self.config["openai_api_key"].get()
        if not api_key:
            raise ValueError("OpenAI API key is required. Please configure it in the settings.")
        
        llm = OpenAILike(
            model=self.config["openai_model"].get(),
            api_base=self.config["api_base"].get(),
            api_key=api_key,
            timeout=120,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000
        )
        
        return llm
    
    def _setup_ui(self):
        """Setup user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Midscene AI Test Code Generator (Bidirectional + Execution)", 
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Configuration area
        config_frame = ttk.LabelFrame(main_frame, text="Configuration")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configuration items
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(config_grid, text="Base URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        base_url_entry = ttk.Entry(config_grid, textvariable=self.config["base_url"], width=50)
        base_url_entry.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(config_grid, text="Test Name:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        test_name_entry = ttk.Entry(config_grid, textvariable=self.config["test_name"], width=50)
        test_name_entry.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(config_grid, text="OpenAI API Base:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        api_base_entry = ttk.Entry(config_grid, textvariable=self.config["api_base"], width=50)
        api_base_entry.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(config_grid, text="OpenAI Model:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        openai_model_entry = ttk.Entry(config_grid, textvariable=self.config["openai_model"], width=50)
        openai_model_entry.grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(config_grid, text="OpenAI API Key:").grid(row=4, column=0, sticky=tk.W, padx=(0, 10))
        api_key_entry = ttk.Entry(config_grid, textvariable=self.config["openai_api_key"], width=50, show="*")
        api_key_entry.grid(row=4, column=1, sticky=tk.W)
        
        # 新增录制配置
        ttk.Label(config_grid, text="Enable Recording:").grid(row=5, column=0, sticky=tk.W, padx=(0, 10))
        recording_checkbox = ttk.Checkbutton(config_grid, variable=self.config["enable_recording"])
        recording_checkbox.grid(row=5, column=1, sticky=tk.W)
        
        ttk.Label(config_grid, text="Recording Output Dir:").grid(row=6, column=0, sticky=tk.W, padx=(0, 10))
        recording_dir_entry = ttk.Entry(config_grid, textvariable=self.config["recording_output_dir"], width=50)
        recording_dir_entry.grid(row=6, column=1, sticky=tk.W)
        
        # Add OpenAI configuration tips
        openai_tips = ttk.Label(
            config_grid, 
            text="💡 Current: gpt-3.5-turbo-instruct | Recommended: 'gpt-4', 'gpt-3.5-turbo-instruct' for better performance",
            font=("Arial", 9),
            foreground="blue"
        )
        openai_tips.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Create input area - using left-right split
        input_container = ttk.Frame(main_frame)
        input_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Left side: code input
        code_input_frame = ttk.LabelFrame(input_container, text="Playwright/Midscene Code Input")
        code_input_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))
        
        self.code_input = scrolledtext.ScrolledText(
            code_input_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=12, 
            font=("Consolas", 10)
        )
        self.code_input.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Right side: natural language input
        nl_input_frame = ttk.LabelFrame(input_container, text="Natural Language Test Description")
        nl_input_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        
        self.natural_language = scrolledtext.ScrolledText(
            nl_input_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=12, 
            font=("Arial", 11)
        )
        self.natural_language.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Example text
        example_text = """Please input "playwright testing framework" in the search box, then press Enter to search.
Wait for the search results to load completely, then check if there are search results containing "playwright".
Click on the first search result, then verify that the page contains relevant content.
Finally, check if the page title contains the "Playwright" keyword."""
        self.natural_language.insert(tk.END, example_text)
        
        # Button area
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row buttons: conversion functions
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        self.code_to_text_btn = ttk.Button(
            button_row1, 
            text="🔄 Code → Natural Language", 
            command=self._convert_code_to_text,
            style="Accent.TButton"
        )
        self.code_to_text_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.text_to_code_btn = ttk.Button(
            button_row1, 
            text="🚀 Natural Language → Midscene Code", 
            command=self._convert_text_to_code,
            style="Accent.TButton"
        )
        self.text_to_code_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.load_example_btn = ttk.Button(
            button_row1, 
            text="📖 Load Example", 
            command=self._load_example
        )
        self.load_example_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(
            button_row1, 
            text="🗑️ Clear All", 
            command=self._clear_all
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Second row buttons: file operations and execution
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill=tk.X)
        
        self.save_btn = ttk.Button(
            button_row2, 
            text="💾 Save Code", 
            command=self._save_code
        )
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.setup_env_btn = ttk.Button(
            button_row2, 
            text="⚙️ Setup Test Environment", 
            command=self._setup_test_environment
        )
        self.setup_env_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.validate_openai_btn = ttk.Button(
            button_row2, 
            text="🔍 Validate OpenAI", 
            command=self._validate_openai_setup
        )
        self.validate_openai_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.force_reinstall_btn = ttk.Button(
            button_row2, 
            text="🔄 Force Reinstall Environment", 
            command=self._force_reinstall_environment
        )
        self.force_reinstall_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.run_test_btn = ttk.Button(
            button_row2, 
            text="▶️ Run Test", 
            command=self._run_test,
            state="disabled"
        )
        self.run_test_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_test_btn = ttk.Button(
            button_row2, 
            text="⏹️ Stop Test", 
            command=self._stop_test,
            state="disabled"
        )
        self.stop_test_btn.pack(side=tk.LEFT)
        
        self.open_report_btn = ttk.Button(
            button_row2, 
            text="📊 Open Latest Report", 
            command=self._open_latest_report
        )
        self.open_report_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Third row buttons: recording controls
        button_row3 = ttk.Frame(button_frame)
        button_row3.pack(fill=tk.X, pady=(5, 0))
        
        self.start_recording_btn = ttk.Button(
            button_row3, 
            text="🎬 Start Recording", 
            command=self._start_recording
        )
        self.start_recording_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_recording_btn = ttk.Button(
            button_row3, 
            text="⏹️ Stop Recording", 
            command=self._stop_recording,
            state="disabled"
        )
        self.stop_recording_btn.pack(side=tk.LEFT)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Generated Midscene AI Test Code (.spec.ts)")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.code_output = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=10, 
            font=("Consolas", 10)
        )
        self.code_output.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Execution output area
        execution_frame = ttk.LabelFrame(main_frame, text="Test Execution Output")
        execution_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.execution_output = scrolledtext.ScrolledText(
            execution_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=8, 
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        self.execution_output.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Status bar
        self.status_bar = ttk.Label(
            main_frame, 
            text="🚀 Ready - Configure settings and generate Midscene AI test code", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _convert_code_to_text(self):
        """Convert code to natural language description"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please input Playwright/Midscene code")
            return
        
        try:
            self.status_bar.config(text="Analyzing code and converting to natural language...")
            self.root.update()
            
            # Create LLM with validation
            llm = self._create_code_to_text_chain()
            
            # Create system and user messages
            system_prompt = """
You are a professional Playwright code interpreter. Convert Python Playwright code into clear, concise natural language instructions.

IMPORTANT RULES:
1. Output ONLY the step-by-step test actions
2. Do NOT include any metadata, token usage, or API response information
3. Do NOT include setup code like imports or test fixtures
4. Use numbered list format (1. 2. 3. etc.)
5. Include element locators when present
6. Do not include code syntax or markdown

Example input:
page.goto("https://example.com");
page.locator("button").click();

Example output:
1. Navigate to "https://example.com"
2. Click the button element

Convert the following code to natural language instructions:
            """
            
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=code)
            ]
            
            print(f"Debug - Input code sent to LLM: {code[:100]}...")
            result = llm.chat(messages)
            print(f"Debug - LLM response type: {type(result)}")
            print(f"Debug - LLM response content: {str(result)[:200]}...")
            
            # Clear and update natural language description area
            self.natural_language.delete("1.0", tk.END)
            # Extract content from ChatResponse
            if hasattr(result, 'message') and hasattr(result.message, 'content'):
                self.natural_language.insert(tk.END, result.message.content.strip())
            else:
                self.natural_language.insert(tk.END, str(result).strip())
            self.status_bar.config(text="✅ Code successfully converted to natural language description")
            
        except ValueError as ve:
            messagebox.showerror("Configuration Error", str(ve))
            self.status_bar.config(text="❌ Configuration error")
        except Exception as e:
            error_msg = str(e)
            if "500" in error_msg or "Internal Server Error" in error_msg:
                messagebox.showerror("API Error", f"API server error (HTTP 500). This may be due to:\n\n1. Invalid API key\n2. Unsupported model name\n3. API service temporarily unavailable\n\nPlease check your configuration and try again.\n\nError details: {error_msg}")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                messagebox.showerror("Authentication Error", "API key is invalid or expired. Please check your OpenAI API key.")
            elif "timeout" in error_msg.lower():
                messagebox.showerror("Timeout Error", "API request timed out. Please check your network connection and try again.")
            else:
                messagebox.showerror("Error", f"Error converting code: {error_msg}")
            self.status_bar.config(text="❌ Code conversion failed")
            print(f"Debug - Code to text error: {e}")
    
    def _convert_text_to_code(self):
        """Generate Midscene AI test code"""
        text = self.natural_language.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please input natural language test description")
            return
        
        try:
            self.status_bar.config(text="Generating Midscene AI test code...")
            self.root.update()
            
            # Prepare template parameters
            base_url = self.config["base_url"].get()
            test_name = self.config["test_name"].get()
            
            # Debug information
            print(f"Debug - Base URL: '{base_url}'")
            print(f"Debug - Test Name: '{test_name}'")
            print(f"Debug - Description: '{text[:50]}...'")
            
            # Validate parameters
            if not base_url:
                messagebox.showwarning("Warning", "Please configure Base URL")
                return
            
            if not test_name:
                messagebox.showwarning("Warning", "Please configure Test Name")
                return
            
            # Create LLM with validation
            llm = self._create_text_to_code_chain()
            
            # Read API.md content to include in prompt
            api_content = ""
            try:
                # Try multiple possible locations for API.md
                api_paths = [
                    os.path.join(PROJECT_DIR, "API.md"),
                    os.path.join(os.getcwd(), "API.md"),
                    os.path.join(SCRIPT_DIR, "API.md")
                ]
                
                for api_path in api_paths:
                    if os.path.exists(api_path):
                        with open(api_path, "r", encoding="utf-8") as f:
                            api_content = f.read()
                        print(f"Debug - Successfully loaded API.md from: {api_path}")
                        break
                else:
                    print("Warning: API.md not found in any expected location")
                    api_content = "Basic Midscene AI API reference not available"
            except Exception as e:
                print(f"Warning: Error reading API.md: {e}")
                api_content = "Basic Midscene AI API reference not available"
            
            # Build system prompt
            system_prompt = f"""
You are a professional Midscene AI test code generation expert. Generate high-quality TypeScript test code based on natural language descriptions.

### Complete Midscene AI API Documentation:
{api_content}

### Important Rules:
1. **Strictly follow Midscene AI API specifications from the documentation above**
2. **Generated code must be directly executable**
3. **Use correct type definitions and import statements**
4. **Include reasonable waits and assertions**
5. **All code comments and strings must be in English**
6. **Always use natural language descriptions for element location, NOT CSS selectors or XPath**
7. **Use proper aiAssert syntax: await aiAssert('natural language assertion')**
8. **DO NOT include page.goto() or page.setViewportSize() in test function - these are in beforeEach**
9. **DO NOT use aiAction() for navigation - page navigation is handled in beforeEach**
10. **Start test steps directly with UI interactions like aiTap, aiInput, etc.**

### CRITICAL: Element Description Requirements for VLM Recognition:
**All element descriptions MUST be highly specific and detailed to help Vision Language Models accurately identify UI elements:**

1. **Include Visual Characteristics:**
   - Color, size, shape, position relative to other elements
   - Text content, icons, or visual indicators
   - Background color, border style, or visual styling

2. **Include Contextual Information:**
   - Location on page (top-left, center, bottom-right, etc.)
   - Relationship to nearby elements
   - Section or container the element belongs to

3. **Include Functional Information:**
   - Element type (button, input field, dropdown, link, etc.)
   - Purpose or action the element performs
   - State information (enabled/disabled, selected/unselected)

### Element Description Examples:
**❌ BAD (too vague):**
- await aiTap('login button')
- await aiInput('username', 'username field')
- await aiAssert('form is submitted')

**✅ GOOD (specific and detailed):**
- await aiTap('blue rectangular login button with white text located at the bottom-right of the login form')
- await aiInput('testuser@example.com', 'username input field with gray border and placeholder text "Enter your email" located at the top of the login form')
- await aiAssert('green success message "Login successful" appears at the top of the page with checkmark icon')

**✅ EXCELLENT (highly detailed for complex scenarios):**
- await aiTap('red circular delete button with trash can icon located in the top-right corner of the user profile card with white background')
- await aiInput('New Product Name', 'product name input field with blue focus border, located in the first row of the product creation form under the "Basic Information" section header')
- await aiAssert('shopping cart icon in the top navigation bar shows red badge with number "3" indicating three items in cart')

### Key API Functions with Enhanced Description Requirements:
- aiTap('detailed element description with visual and positional context'): Click/tap elements
- aiInput('text', 'detailed input field description with visual characteristics and location'): Input text into elements
- aiQuery({{key: 'detailed description of data to extract with visual context'}}): Extract data from UI
- aiAssert('detailed assertion description with specific visual indicators and expected states'): Natural language assertions
- aiWaitFor('detailed condition description with specific visual elements to wait for'): Wait for conditions
- aiHover('detailed element description with visual and contextual information'): Hover over elements
- aiKeyboardPress('key', 'detailed element description if targeting specific element'): Press keyboard keys
- aiScroll({{direction: 'up/down/left/right'}}, 'detailed description of scrollable area or container'): Scroll
- ai('complex multi-step action description with detailed element specifications'): For complex actions only

### Standard Template Structure:
```typescript
import {{ test as base }} from '@playwright/test';
import type {{ PlayWrightAiFixtureType }} from '@midscene/web/playwright';
import {{ PlaywrightAiFixture }} from '@midscene/web/playwright';

export const test = base.extend<PlayWrightAiFixtureType>(PlaywrightAiFixture({{
  waitForNetworkIdleTimeout: 2000,
  ignoreHTTPSErrors: true,
}}));

test.beforeEach(async ({{ page }}) => {{
  await page.goto('{base_url}');
  await page.setViewportSize({{ width: 1440, height: 900 }});
}});

test('{test_name}', async ({{
  aiAction,
  aiTap,
  aiInput,
  aiKeyboardPress,
  aiHover,
  aiWaitFor,
  aiQuery,
  aiAssert,
  ai,
  page,
}}) => {{
  // Test steps will be generated here
}});
```
"""
            
            # Build user prompt
            user_prompt = f"""
Generate a complete Midscene AI test file based on this description:

Base URL: {base_url}
Test Name: {test_name}
Test Description: {text}

IMPORTANT CONSTRAINTS:
1. DO NOT use aiAction() for page navigation or opening URLs - this is handled in beforeEach
2. DO NOT include any Navigate actions in the test
3. Start the test directly with UI interactions like aiTap, aiInput, etc.
4. The page is already loaded and ready when the test starts
5. Focus on the actual UI interactions described in the test description
6. **ONLY OUTPUT THE MIDSCENE TEST CODE - DO NOT include any explanations, comments, or additional descriptions**
7. **DO NOT add any text before or after the code - just return the pure TypeScript code**
8. **DO NOT include markdown code blocks or formatting - return raw code only**

Please generate complete, directly executable Midscene AI test code following the template structure above.
"""
            
            # Create messages for chat API
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ]
            
            print(f"Debug - Sending to LLM: {user_prompt[:100]}...")
            
            # Call LLM with chat API (consistent with code_to_text)
            result = llm.chat(messages)
            print(f"Debug - LLM response type: {type(result)}")
            
            # Extract content from ChatResponse
            if hasattr(result, 'message') and hasattr(result.message, 'content'):
                cleaned_code = self._clean_generated_code(result.message.content)
            else:
                cleaned_code = self._clean_generated_code(str(result))
            
            self.code_output.delete("1.0", tk.END)
            self.code_output.insert(tk.END, cleaned_code)
            self.status_bar.config(text="✅ Midscene AI test code generation completed")
            
        except ValueError as ve:
            messagebox.showerror("Configuration Error", str(ve))
            self.status_bar.config(text="❌ Configuration error")
        except Exception as e:
            error_msg = str(e)
            if "500" in error_msg or "Internal Server Error" in error_msg:
                messagebox.showerror("API Error", f"API server error (HTTP 500). This may be due to:\n\n1. Invalid API key\n2. Unsupported model name\n3. API service temporarily unavailable\n\nPlease check your configuration and try again.\n\nError details: {error_msg}")
            elif "401" in error_msg or "Unauthorized" in error_msg:
                messagebox.showerror("Authentication Error", "API key is invalid or expired. Please check your OpenAI API key.")
            elif "timeout" in error_msg.lower():
                messagebox.showerror("Timeout Error", "API request timed out. Please check your network connection and try again.")
            else:
                messagebox.showerror("Error", f"Error generating code: {error_msg}")
            self.status_bar.config(text="❌ Code generation failed")
            print(f"Debug - Error details: {e}")
    
    def _clean_generated_code(self, code):
        """Clean generated code"""
        # Remove common LLM response prefixes
        prefixes = ["```typescript", "```ts", "```javascript", "```js", "```", "// Code:", "// Output:"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
        
        if code.endswith("```"):
            code = code[:-3].rstrip()
        
        # Import check removed as requested
        
        return code
    
    def _save_code(self):
        """Save generated code to file"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to save")
            return
        
        try:
            from tkinter import filedialog
            
            # Generate default filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            test_name = self.config['test_name'].get().strip()
            if test_name:
                # Use test name as part of filename (sanitize for filesystem)
                safe_test_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_test_name = safe_test_name.replace(' ', '_')
                default_filename = f"midscene_test_{timestamp}_{safe_test_name}.spec.ts"
            else:
                default_filename = f"midscene_test_{timestamp}.spec.ts"
            
            # Try using the simplified file dialog first
            try:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".spec.ts",
                    title="Save Midscene Test Code",
                    initialfile=default_filename
                )
            except Exception as dialog_error:
                print(f"Debug - File dialog error: {dialog_error}")
                # Fallback: save to current directory with timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"midscene_test_{timestamp}.spec.ts"
                messagebox.showinfo("Info", f"File dialog error, saving to current directory as: {filename}")
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                self.saved_file_path = filename
                
                # Create report directory in the same location as the saved file
                file_dir = os.path.dirname(os.path.abspath(filename))
                self.report_dir = os.path.join(file_dir, "midscene_run", "report")
                os.makedirs(self.report_dir, exist_ok=True)
                
                messagebox.showinfo("Success", f"Code saved to: {filename}\nReports will be saved to: {self.report_dir}")
                self.status_bar.config(text=f"✅ Code saved to: {filename}")
                self._log_execution(f"📁 Code saved to: {filename}")
                self._log_execution(f"📊 Reports will be saved to: {self.report_dir}")
                
                # Enable execution related buttons after saving
                self.run_test_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")
            print(f"Debug - Save error details: {e}")
    
    def _setup_test_environment(self):
        """Setup test environment"""
        try:
            # First validate OpenAI service
            self.status_bar.config(text="Validating OpenAI service...")
            self.root.update()
            
            is_openai_ok, openai_message = self._check_openai_service()
            if not is_openai_ok:
                self._log_execution(openai_message)
                result = messagebox.askyesno(
                    "OpenAI Service Issue", 
                    f"{openai_message}\n\nDo you want to continue setting up the environment anyway?\n\n"
                    f"To fix this issue:\n"
                    f"1. Set your OpenAI API key\n"
                    f"2. Ensure you have access to the model: {self.config['openai_model'].get()}\n"
                    f"3. Check your API base URL: {self.config['api_base'].get()}"
                )
                if not result:
                    self.status_bar.config(text="❌ Environment setup cancelled - Please fix OpenAI configuration first")
                    return
            else:
                self._log_execution(openai_message)
            
            self.status_bar.config(text="Setting up test environment...")
            self.root.update()
            
            # Check if environment already exists and is valid
            if self._check_existing_environment():
                self._log_execution("✅ Found existing test environment, reusing...")
                self._log_execution(f"📁 Test directory: {self.temp_test_dir}")
                self.status_bar.config(text="✅ Test environment ready (reused existing)")
                self.setup_env_btn.config(state="normal")
                self.run_test_btn.config(state="normal")
                return
            
            # Create new temporary test directory
            if self.temp_test_dir and os.path.exists(self.temp_test_dir):
                import shutil
                shutil.rmtree(self.temp_test_dir)
            
            self.temp_test_dir = tempfile.mkdtemp(prefix="midscene_test_")
            
            # Create package.json
            package_json = {
                "name": "midscene-test",
                "version": "1.0.0",
                "description": "Generated Midscene test",
                "scripts": {
                    "test": "playwright test",
                    "test:headed": "playwright test --headed",
                    "test:debug": "playwright test --debug"
                },
                "devDependencies": {
                    "@midscene/web": "^0.20.0",
                    "@playwright/test": "^1.53.1",
                    "@types/node": "^24.0.3"
                },
                "dependencies": {
                    "dotenv": "^16.4.7"
                }
            }
            
            with open(os.path.join(self.temp_test_dir, "package.json"), 'w', encoding='utf-8') as f:
                json.dump(package_json, f, indent=2)
            
            # Create .env file for Midscene configuration with OpenAI
            env_content = f"""# Midscene AI Configuration for OpenAI
OPENAI_API_KEY={self.config["openai_api_key"].get()}
OPENAI_BASE_URL={self.config["api_base"].get()}
MIDSCENE_MODEL_NAME={self.config["openai_model"].get()}

# Debug settings (optional)
# DEBUG=midscene:ai:profile:stats
# DEBUG=midscene:ai:call

# Additional OpenAI-specific settings
# MIDSCENE_PREFERRED_LANGUAGE=English
"""
            
            with open(os.path.join(self.temp_test_dir, ".env"), 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            # Create playwright.config.ts with dynamic report output path
            playwright_config = '''import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';
import path from 'path';

// Load .env file
dotenv.config();

export default defineConfig({
  testDir: './tests',
  timeout: 90 * 1000,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ["list"], 
    ["@midscene/web/playwright-report", {
      outputFolder: path.join(process.cwd(), 'midscene_run', 'report')
    }]
  ],
  use: {
    trace: 'on-first-retry',
    ignoreHTTPSErrors: true,
    // 启用CDP调试端口以支持录制连接
    launchOptions: {
      args: ['--remote-debugging-port=9222']
    }
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
'''
            
            with open(os.path.join(self.temp_test_dir, "playwright.config.ts"), 'w', encoding='utf-8') as f:
                f.write(playwright_config)
            
            # Create tests directory
            tests_dir = os.path.join(self.temp_test_dir, "tests")
            os.makedirs(tests_dir, exist_ok=True)
            
            self._log_execution("✅ Test environment setup completed")
            self._log_execution(f"📁 Test directory: {self.temp_test_dir}")
            self._log_execution(f"🤖 OpenAI model: {self.config['openai_model'].get()}")
            self._log_execution(f"🔗 API endpoint: {self.config['api_base'].get()}")
            
            # Install dependencies in background
            threading.Thread(target=self._install_dependencies, daemon=True).start()
            
            self.status_bar.config(text="✅ Test environment setup completed, installing dependencies...")
            self.setup_env_btn.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error setting up test environment: {str(e)}")
            self.status_bar.config(text="❌ Test environment setup failed")
    
    def _check_existing_environment(self):
        """Check if existing test environment is valid and can be reused"""
        if not self.temp_test_dir or not os.path.exists(self.temp_test_dir):
            return False
        
        try:
            # Check required files exist
            required_files = [
                "package.json",
                "playwright.config.ts", 
                ".env"
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(self.temp_test_dir, file)):
                    return False
            
            # Check if node_modules exists (dependencies installed)
            node_modules_path = os.path.join(self.temp_test_dir, "node_modules")
            if not os.path.exists(node_modules_path):
                return False
            
            # Check if key packages are installed
            key_packages = ["@midscene/web", "@playwright/test"]
            for package in key_packages:
                package_path = os.path.join(node_modules_path, package)
                if not os.path.exists(package_path):
                    return False
            
            # Check if tests directory exists
            tests_dir = os.path.join(self.temp_test_dir, "tests")
            if not os.path.exists(tests_dir):
                os.makedirs(tests_dir, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Debug - Environment check error: {e}")
            return False
    
    def _get_npm_command(self):
        """Get npm command based on operating system"""
        import platform
        if platform.system() == "Windows":
            return "npm.cmd"
        return "npm"
    
    def _get_npx_command(self):
        """Get npx command based on operating system"""
        import platform
        if platform.system() == "Windows":
            return "npx.cmd"
        return "npx"
    
    def _install_dependencies(self):
        """Install dependencies in background"""
        try:
            import platform
            self._log_execution("📦 Installing Node.js dependencies...")
            
            # Install dependencies
            npm_cmd = self._get_npm_command()
            result = subprocess.run(
                [npm_cmd, "install"],
                cwd=self.temp_test_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                shell=True if platform.system() == "Windows" else False
            )
            
            if result.returncode == 0:
                self._log_execution("✅ Dependencies installed successfully")
                
                # Install Playwright browsers
                self._log_execution("🌐 Installing Playwright browsers...")
                npx_cmd = self._get_npx_command()
                browser_result = subprocess.run(
                    [npx_cmd, "playwright", "install", "chromium"],
                    cwd=self.temp_test_dir,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes timeout
                    shell=True if platform.system() == "Windows" else False
                )
                
                if browser_result.returncode == 0:
                    self._log_execution("✅ Playwright browsers installed successfully")
                    self.root.after(0, lambda: self.status_bar.config(text="✅ Test environment fully ready"))
                    self.root.after(0, lambda: self.setup_env_btn.config(state="normal"))
                else:
                    self._log_execution(f"❌ Browser installation failed: {browser_result.stderr}")
            else:
                self._log_execution(f"❌ Dependencies installation failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self._log_execution("⏰ Installation timeout, please check network connection")
        except Exception as e:
            self._log_execution(f"❌ Installation error: {str(e)}")
    
    def _run_test(self):
        """Execute test"""
        if not self.temp_test_dir:
            messagebox.showwarning("Warning", "Please setup test environment first")
            return
        
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to execute")
            return
        
        # Check if recording is enabled and prompt user
        if self.config["enable_recording"].get() and not self.recording_process:
            result = messagebox.askyesno(
                "Recording Available", 
                "Recording is enabled but not started.\n\n"
                "Would you like to start recording before running the test?\n\n"
                "This will help you capture user interactions for future test generation."
            )
            if result:
                self._start_recording()
                # Give user time to start recording
                messagebox.showinfo(
                    "Recording Started", 
                    "Recording has been started!\n\n"
                    "You can now interact with the browser to record actions.\n"
                    "Click OK to continue with test execution."
                )
        
        # Log recording status
        if self.recording_process:
            self._log_execution("🎥 Recording is active during test execution")
        
        try:
            # Generate default filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            test_name = self.config['test_name'].get().strip()
            if test_name:
                # Use test name as part of filename (sanitize for filesystem)
                safe_test_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_test_name = safe_test_name.replace(' ', '_')
                default_filename = f"midscene_test_{timestamp}_{safe_test_name}.spec.ts"
            else:
                default_filename = f"midscene_test_{timestamp}.spec.ts"
            
            # Always save and test only the generated code in temp directory
            test_file = os.path.join(self.temp_test_dir, "tests", default_filename)
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Set report directory in temp test directory
            self.report_dir = os.path.join(self.temp_test_dir, "midscene_run", "report")
            os.makedirs(self.report_dir, exist_ok=True)
            
            self._log_execution(f"📄 Generated test file: {default_filename}")
            self._log_execution(f"📁 Test file saved to: {test_file}")
            self._log_execution("🚀 Starting test execution with generated code only...")
            self._log_execution(f"📊 Reports will be saved to: {self.report_dir}")
            
            # Execute test in new thread
            threading.Thread(target=self._execute_playwright_test, args=(test_file,), daemon=True).start()
            
            # Update button states
            self.run_test_btn.config(state="disabled")
            self.stop_test_btn.config(state="normal")
            self.status_bar.config(text="▶️ Executing test...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error executing test: {str(e)}")
            self.status_bar.config(text="❌ Test execution failed")
    
    def _execute_playwright_test(self, test_pattern):
        """Execute Playwright test"""
        try:
            import platform
            import shutil
            # Set up environment variables
            test_env = os.environ.copy()
            # Set ignore HTTPS errors for temporary test environment
            test_env['PLAYWRIGHT_IGNORE_HTTPS_ERRORS'] = '1'
            
            # Always use temp_test_dir as working directory
            work_dir = self.temp_test_dir
            
            # Convert absolute path to relative path for Windows compatibility
            if os.path.isabs(test_pattern):
                # Get relative path from work_dir
                try:
                    relative_test_pattern = os.path.relpath(test_pattern, work_dir)
                    # On Windows, use forward slashes for better compatibility
                    if platform.system() == "Windows":
                        relative_test_pattern = relative_test_pattern.replace(os.sep, '/')
                    test_pattern = relative_test_pattern
                except ValueError:
                    # If relative path calculation fails, use just the filename
                    test_pattern = os.path.basename(test_pattern)
            
            # Always run the specific generated test file
            test_cmd = [self._get_npx_command(), "playwright", "test", test_pattern, "--headed"]
            
            # Execute test with custom report location
            self.execution_process = subprocess.Popen(
                test_cmd,
                cwd=work_dir,
                env=test_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                shell=True if platform.system() == "Windows" else False
            )
            
            # Read output in real time
            for line in iter(self.execution_process.stdout.readline, ''):
                if line:
                    self.root.after(0, lambda l=line: self._log_execution(l.strip()))
            
            # Wait for process to end
            return_code = self.execution_process.wait()
            
            if return_code == 0:
                self.root.after(0, lambda: self._log_execution("✅ Test execution successful"))
                self.root.after(0, lambda: self._log_execution(f"📊 Test report saved to: {self.report_dir}"))
                
                # Copy report to project root directory for persistent access
                self.root.after(0, self._copy_report_to_project)
                
                self.root.after(0, lambda: self.status_bar.config(text="✅ Test execution completed"))
                # Try to open report folder
                self.root.after(0, self._show_report_location)
            else:
                self.root.after(0, lambda: self._log_execution(f"❌ Test execution failed, exit code: {return_code}"))
                self.root.after(0, lambda: self.status_bar.config(text="❌ Test execution failed"))
            
        except Exception as e:
            self.root.after(0, lambda: self._log_execution(f"❌ Test execution error: {str(e)}"))
            self.root.after(0, lambda: self.status_bar.config(text="❌ Test execution error"))
        finally:
            # Reset button states
            self.root.after(0, lambda: self.run_test_btn.config(state="normal"))
            self.root.after(0, lambda: self.stop_test_btn.config(state="disabled"))
            self.execution_process = None
    
    def _copy_report_to_project(self):
        """Copy report from temp directory to project root directory"""
        try:
            if hasattr(self, 'report_dir') and self.report_dir and os.path.exists(self.report_dir):
                # Create project report directory
                project_report_dir = os.path.join(PROJECT_DIR, "midscene_run", "report")
                os.makedirs(project_report_dir, exist_ok=True)
                
                # Copy all files from temp report dir to project report dir
                import shutil
                latest_report_file = None
                latest_time = 0
                
                for item in os.listdir(self.report_dir):
                    src_path = os.path.join(self.report_dir, item)
                    dst_path = os.path.join(project_report_dir, item)
                    
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                        # Track the latest HTML report file
                        if item.endswith('.html'):
                            file_time = os.path.getmtime(src_path)
                            if file_time > latest_time:
                                latest_time = file_time
                                latest_report_file = dst_path
                    elif os.path.isdir(src_path):
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
                
                self._log_execution(f"📋 Report saved to project directory: {project_report_dir}")
                
                # Create a convenient shortcut to the latest report
                if latest_report_file:
                    shortcut_path = os.path.join(PROJECT_DIR, "latest_test_report.html")
                    try:
                        if os.path.exists(shortcut_path):
                            os.remove(shortcut_path)
                        shutil.copy2(latest_report_file, shortcut_path)
                        self._log_execution(f"📄 Latest report copied to: {shortcut_path}")
                        self._log_execution(f"💡 Tip: Double-click 'latest_test_report.html' in project root to view the report")
                    except Exception as e:
                        self._log_execution(f"⚠️ Warning: Failed to create report shortcut: {str(e)}")
                
                # Update report_dir to point to project directory for opening
                self.project_report_dir = project_report_dir
                
                # List all report files for user reference
                report_files = [f for f in os.listdir(project_report_dir) if f.endswith('.html')]
                if report_files:
                    self._log_execution(f"📊 Available report files ({len(report_files)}):")
                    for i, report_file in enumerate(sorted(report_files, reverse=True)[:3]):  # Show latest 3
                        self._log_execution(f"   {i+1}. {report_file}")
                    if len(report_files) > 3:
                        self._log_execution(f"   ... and {len(report_files) - 3} more files")
                
        except Exception as e:
            self._log_execution(f"⚠️ Warning: Failed to copy report to project directory: {str(e)}")
    
    def _stop_test(self):
        """Stop test execution"""
        if self.execution_process:
            try:
                self.execution_process.terminate()
                self._log_execution("⏹️ Test stopped")
                self.status_bar.config(text="⏹️ Test stopped")
            except Exception as e:
                self._log_execution(f"❌ Error stopping test: {str(e)}")
            finally:
                self.run_test_btn.config(state="normal")
                self.stop_test_btn.config(state="disabled")
                self.execution_process = None
        
        # Also stop recording if it's active
        if self.recording_process:
            try:
                self._log_execution("🎥 Stopping recording process...")
                self._stop_recording()
            except Exception as e:
                self._log_execution(f"❌ Error stopping recording: {str(e)}")
    
    def _show_report_location(self):
        """Show report location to user"""
        # Prefer project report directory if available
        report_path = getattr(self, 'project_report_dir', None) or getattr(self, 'report_dir', None)
        
        if report_path and os.path.exists(report_path):
            # Check for latest_test_report.html in project root
            latest_report_shortcut = os.path.join(PROJECT_DIR, "latest_test_report.html")
            
            # Show both locations if project report exists
            message = f"🎉 Test execution completed!\n\n"
            
            if os.path.exists(latest_report_shortcut):
                message += f"📄 Quick Access: 'latest_test_report.html' in project root\n"
                message += f"📁 Full reports folder: {report_path}\n\n"
                message += f"💡 You can:\n"
                message += f"   • Double-click 'latest_test_report.html' for latest results\n"
                message += f"   • Browse '{os.path.basename(report_path)}' folder for all reports\n\n"
            else:
                if hasattr(self, 'project_report_dir') and self.project_report_dir:
                    message += f"📁 Report saved to project directory:\n{self.project_report_dir}\n\n"
                    if hasattr(self, 'report_dir') and self.report_dir != self.project_report_dir:
                        message += f"📁 Temp report location:\n{self.report_dir}\n\n"
                else:
                    message += f"📁 Report saved to:\n{report_path}\n\n"
            
            message += "Do you want to open the report folder now?"
            
            result = messagebox.askyesno("Test Report", message)
            if result:
                try:
                    import platform
                    system = platform.system()
                    # Try to open the project root directory first if shortcut exists
                    open_path = PROJECT_DIR if os.path.exists(latest_report_shortcut) else report_path
                    
                    if system == "Darwin":  # macOS
                        subprocess.run(["open", open_path])
                    elif system == "Windows":
                        subprocess.run(["explorer", open_path])
                    elif system == "Linux":
                        subprocess.run(["xdg-open", open_path])
                        
                    # Additional message about the shortcut
                    if os.path.exists(latest_report_shortcut):
                        self._log_execution(f"📂 Opened project directory. Look for 'latest_test_report.html'")
                        
                except Exception as e:
                    self._log_execution(f"❌ Error opening report folder: {str(e)}")
        else:
            messagebox.showwarning("No Report", "No test report found. Please run a test first.")
            self._log_execution("❌ No test report found to display")
    
    def _open_latest_report(self):
        """Open the latest test report"""
        try:
            # First check for the shortcut in project root
            latest_report_shortcut = os.path.join(PROJECT_DIR, "latest_test_report.html")
            
            if os.path.exists(latest_report_shortcut):
                # Open the shortcut file directly
                self._open_file_in_browser(latest_report_shortcut)
                self._log_execution(f"📊 Opened latest test report: latest_test_report.html")
                return
            
            # If no shortcut, look in the project report directory
            project_report_dir = os.path.join(PROJECT_DIR, "midscene_run", "report")
            if os.path.exists(project_report_dir):
                html_files = [f for f in os.listdir(project_report_dir) if f.endswith('.html')]
                if html_files:
                    # Find the most recent HTML file
                    latest_file = None
                    latest_time = 0
                    for html_file in html_files:
                        file_path = os.path.join(project_report_dir, html_file)
                        file_time = os.path.getmtime(file_path)
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_file = file_path
                    
                    if latest_file:
                        self._open_file_in_browser(latest_file)
                        self._log_execution(f"📊 Opened latest test report: {os.path.basename(latest_file)}")
                        return
            
            # No reports found
            messagebox.showinfo("No Report", 
                              "No test reports found.\n\n"
                              "Please run a test first to generate a report.\n"
                              "Reports will be saved to 'midscene_run/report/' directory.")
            self._log_execution("❌ No test reports found to open")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error opening report: {str(e)}")
            self._log_execution(f"❌ Error opening latest report: {str(e)}")
    
    def _open_file_in_browser(self, file_path):
        """Open a file in the default browser"""
        import platform
        import webbrowser
        
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(file_path)
            
            # Use webbrowser module for cross-platform compatibility
            webbrowser.open(f"file://{abs_path}")
            
        except Exception as e:
            # Fallback to system-specific commands
            system = platform.system()
            try:
                if system == "Darwin":  # macOS
                    subprocess.run(["open", file_path])
                elif system == "Windows":
                    subprocess.run(["start", file_path], shell=True)
                elif system == "Linux":
                    subprocess.run(["xdg-open", file_path])
            except Exception as e2:
                raise Exception(f"Failed to open file: {str(e)} (fallback also failed: {str(e2)})")
    
    def _log_execution(self, message):
        """Log execution messages"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.execution_output.insert(tk.END, log_message)
        self.execution_output.see(tk.END)
        self.root.update_idletasks()
    
    def _load_example(self):
        """Load example data"""
        # Clear all fields
        self._clear_all()
        
        # Set example configuration
        self.config["base_url"].set("https://cn.bing.com")
        self.config["test_name"].set("search playwright on bing")
        
        # Load example code with correct Midscene API usage
        example_code = """import { test as base } from '@playwright/test';
import type { PlayWrightAiFixtureType } from '@midscene/web/playwright';
import { PlaywrightAiFixture } from '@midscene/web/playwright';

export const test = base.extend<PlayWrightAiFixtureType>(PlaywrightAiFixture({
  waitForNetworkIdleTimeout: 2000,
  ignoreHTTPSErrors: true,
}));

test.beforeEach(async ({ page }) => {
  await page.goto('https://cn.bing.com');
  await page.setViewportSize({ width: 1440, height: 900 });
});

test('search playwright on bing', async ({
  aiInput,
  aiAssert,
  aiQuery,
  aiTap,
  aiWaitFor,
  aiKeyboardPress,
  page,
}) => {
  // Input search keyword using natural language description
  await aiInput('playwright', 'search input box');
  
  // Press Enter to execute search
  await aiKeyboardPress('Enter');
  
  // Wait for search results to load
  await aiWaitFor('search results are displayed');
  
  // Extract search results using natural language descriptions
  const searchResults = await aiQuery({
    titles: 'search result titles as string array',
    firstResultTitle: 'first search result title as string'
  });
  
  console.log('Search results:', searchResults);
  
  // Verify search results using natural language assertions
  await aiAssert('search results contain playwright related content');
  await aiAssert('there are more than 1 search results');
});"""
        
        # Load example natural language description
        example_text = """Execute the following test steps on the Bing search page:

1. Input the keyword "playwright" in the search box
2. Press Enter to execute the search
3. Wait for the search results page to fully load
4. Extract the title information of search results
5. Verify that search results contain "playwright" related content
6. Verify that the number of search results is greater than 1

Please ensure each step has appropriate waiting and verification mechanisms."""
        
        self.code_input.insert(tk.END, example_code)
        self.natural_language.insert(tk.END, example_text)
        self.status_bar.config(text="📖 Example data loaded - Test bidirectional conversion and execution features")
    
    def _clear_all(self):
        """Clear all text areas"""
        self.code_input.delete("1.0", tk.END)
        self.natural_language.delete("1.0", tk.END)
        self.code_output.delete("1.0", tk.END)
        self.execution_output.delete("1.0", tk.END)
        self.status_bar.config(text="🗑️ All content cleared")
    
    def _force_reinstall_environment(self):
        """Force reinstall test environment (delete existing and create new)"""
        try:
            self.status_bar.config(text="Force reinstalling test environment...")
            self.root.update()
            
            # Delete existing environment
            if self.temp_test_dir and os.path.exists(self.temp_test_dir):
                import shutil
                self._log_execution("🗑️ Removing existing test environment...")
                shutil.rmtree(self.temp_test_dir)
                self.temp_test_dir = None
            
            # Create fresh environment
            self._log_execution("🔄 Creating fresh test environment...")
            self._setup_test_environment()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error force reinstalling environment: {str(e)}")
            self.status_bar.config(text="❌ Force reinstall failed")
    
    def _update_environment_status(self):
        """Update environment status display"""
        if self._check_existing_environment():
            self.status_bar.config(text=f"✅ Test environment ready - {self.temp_test_dir}")
            self.run_test_btn.config(state="normal")
        else:
            self.status_bar.config(text="⚙️ Ready - Click 'Setup Test Environment' to begin")
            self.run_test_btn.config(state="disabled")
    
    def _check_openai_service(self):
        """Check if OpenAI service is accessible"""
        try:
            import requests
            
            api_key = self.config["openai_api_key"].get()
            if not api_key:
                return False, "❌ OpenAI API key not configured"
            
            # Check OpenAI service health
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Try to list models to verify API access
            response = requests.get(f"{self.config['api_base'].get()}/models", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                current_model = self.config["openai_model"].get()
                
                if current_model in models:
                    return True, f"✅ OpenAI service accessible, model '{current_model}' available"
                else:
                    return True, f"✅ OpenAI service accessible, but model '{current_model}' not found in list. It may still work."
            elif response.status_code == 401:
                return False, "❌ OpenAI API key is invalid or expired"
            elif response.status_code == 403:
                return False, "❌ OpenAI API access forbidden. Check your API key permissions."
            else:
                return False, f"❌ OpenAI service error (status: {response.status_code})"
                
        except ImportError:
            return False, "❌ 'requests' library not installed. Run: pip install requests"
        except requests.exceptions.ConnectionError:
            return False, f"❌ Cannot connect to OpenAI at {self.config['api_base'].get()}. Please check your internet connection."
        except requests.exceptions.Timeout:
            return False, "❌ OpenAI service timeout. Please check your internet connection."
        except Exception as e:
            return False, f"❌ OpenAI check failed: {str(e)}"
    
    def _validate_openai_setup(self):
        """Validate OpenAI setup and show status"""
        is_ok, message = self._check_openai_service()
        
        if is_ok:
            messagebox.showinfo("OpenAI Status", message)
            self._log_execution(message)
        else:
            messagebox.showwarning("OpenAI Status", f"{message}\n\nPlease ensure:\n1. OpenAI API key is set correctly\n2. You have access to model: {self.config['openai_model'].get()}\n3. API base URL is correct: {self.config['api_base'].get()}")
            self._log_execution(message)
        
        return is_ok
    
    def _start_recording(self):
        """启动 Playwright codegen 录制"""
        if self.recording_process:
            messagebox.showwarning("Warning", "Recording is already in progress")
            return
        
        try:
            # 创建录制输出目录
            recording_dir = self.config["recording_output_dir"].get()
            if not os.path.isabs(recording_dir):
                recording_dir = os.path.join(PROJECT_DIR, recording_dir)
            os.makedirs(recording_dir, exist_ok=True)
            
            # 生成录制文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file_path = os.path.join(recording_dir, f"recorded_test_{timestamp}.spec.ts")
            
            # 启动 Playwright codegen 连接到现有浏览器
            base_url = self.config["base_url"].get() or "https://example.com"
            cmd = [
                self._get_npx_command(), 
                "playwright", 
                "codegen", 
                "--browser-ws-endpoint=ws://localhost:9222",
                base_url, 
                "--output", 
                self.recording_file_path
            ]
            
            self.recording_process = subprocess.Popen(cmd, cwd=PROJECT_DIR)
            
            # 更新UI状态
            self.start_recording_btn.config(state="disabled")
            self.stop_recording_btn.config(state="normal")
            self.status_bar.config(text=f"🎬 Recording started - Output: {os.path.basename(self.recording_file_path)}")
            self._log_execution(f"🎬 Started Playwright codegen recording (connected to localhost:9222)")
            self._log_execution(f"📁 Recording file: {self.recording_file_path}")
            self._log_execution(f"🌐 Target URL: {base_url}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
            self.status_bar.config(text="❌ Recording start failed")
    
    def _stop_recording(self):
        """停止录制并可选择加载录制的代码"""
        if not self.recording_process:
            messagebox.showwarning("Warning", "No recording in progress")
            return
        
        try:
            # 终止录制进程
            self.recording_process.terminate()
            self.recording_process.wait(timeout=5)
            
            # 更新UI状态
            self.start_recording_btn.config(state="normal")
            self.stop_recording_btn.config(state="disabled")
            self.status_bar.config(text="⏹️ Recording stopped")
            self._log_execution("⏹️ Recording stopped")
            
            # 询问是否加载录制的代码
            if self.recording_file_path and os.path.exists(self.recording_file_path):
                result = messagebox.askyesno(
                    "Load Recorded Code", 
                    f"Recording saved to:\n{self.recording_file_path}\n\nDo you want to load the recorded code?"
                )
                if result:
                    self._load_recorded_code()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")
        finally:
            self.recording_process = None
    
    def _load_recorded_code(self):
        """加载录制的代码到代码输入区域"""
        if not self.recording_file_path or not os.path.exists(self.recording_file_path):
            messagebox.showwarning("Warning", "No recorded code file found")
            return
        
        try:
            with open(self.recording_file_path, 'r', encoding='utf-8') as f:
                recorded_code = f.read()
            
            # 清空并加载录制的代码
            self.code_input.delete("1.0", tk.END)
            self.code_input.insert(tk.END, recorded_code)
            
            self._log_execution(f"📄 Loaded recorded code from: {os.path.basename(self.recording_file_path)}")
            self.status_bar.config(text="✅ Recorded code loaded")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load recorded code: {str(e)}")

def main():
    """Main function"""
    root = tk.Tk()
    app = MidsceneCodeGenerator(root)
    
    # Set window icon (if available)
    try:
        # root.iconbitmap("icon.ico")  # If icon file exists
        pass
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    main()
