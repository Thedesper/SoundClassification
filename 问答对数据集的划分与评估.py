import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import subprocess
import tempfile
import os
import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Model configuration
MODEL_NAME = "qwen3:32b-q8_0"
API_BASE = "http://163.184.132.210:11434"

class PlaywrightCodeInterpreter:
    def __init__(self, root):
        self.root = root
        self.root.title("Playwright Code Interpreter")
        self.root.geometry("1200x900")
        
        # Configuration options
        self.config = {
            "headless": tk.BooleanVar(value=False),
            "model_name": tk.StringVar(value=MODEL_NAME),
            "api_base": tk.StringVar(value=API_BASE),
            "timeout": tk.IntVar(value=60)
        }
        
        # Set theme
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        
        # Create LLM chains with enhanced prompt
        self.code_to_text_chain = self._create_code_to_text_chain()
        self.text_to_code_chain = self._create_text_to_code_chain()
        
        self._setup_ui()
        self._setup_config_panel()
        
    def _create_code_to_text_chain(self):
        """Create LLM chain for code to natural language conversion"""
        llm = Ollama(
            model=self.config["model_name"].get(),
            base_url=self.config["api_base"].get(),
            timeout=self.config["timeout"].get()
        )
        
        template = """
        /no_think You are a professional Playwright code interpreter. Convert Python Playwright code into clear, concise natural language instructions.
        
        Output format:
        1. Step-by-step description of actions
        2. Include element locators (e.g., "[data-test='login-button']")
        3. Do not include code or markdown syntax
        
        Example input:
        page.goto("https://example.com");
        page.locator("button").click();
        
        Example output:
        1. Navigate to "https://example.com"
        2. Click the button element
        
        INPUT: {text}
        OUTPUT: Natural language instructions
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        return LLMChain(llm=llm, prompt=chat_prompt)
    
    def _create_text_to_code_chain(self):
        """Create LLM chain for natural language to code conversion with robust error handling"""
        llm = Ollama(
            model=self.config["model_name"].get(),
            base_url=self.config["api_base"].get(),
            timeout=self.config["timeout"].get()
        )
        
        template = """
        /no_think You are a professional Playwright code generator. Convert natural language to robust Python automation code with the following guidelines:

        ### Robustness & Error Handling Requirements:
        1. **Lifecycle Management**:
           - Enclose all operations in try-finally to ensure browser closure
           - Add `input("Press Enter to close...")` before closing for manual inspection
           - Never call `context.close()` or `browser.close()` prematurely
        
        2. **Error Handling**:
           - Wrap every critical action (click/fill/navigate) in try-except blocks
           - Log errors with timestamps and context (e.g., `print(f"Error at step X: {e}")`)
           - Capture screenshots on error: `page.screenshot(path="error-stepX.png")`
           - Use specific exceptions (TimeoutError, ElementNotFoundError) when possible
        
        3. **Element Validation**:
           - Always validate elements before interaction:
             `expect(element).to_be_visible()`, `expect(element).to_be_enabled()`
           - Use explicit waits with states: `wait_for_selector(state="visible", timeout=10000)`
           - Prefer element-based waits over `time.sleep()`
        
        4. **Locator Best Practices**:
           - Prioritize semantic selectors: `get_by_text`, `get_by_role`, `get_by_label`
           - Use `exact=True` for text selectors when ambiguity exists
           - Avoid index-based selectors (.first, .nth) unless necessary
           - Prefer unique attributes: `data-test`, `id`, `name`
        
        5. **Debugging Aids**:
           - Set `slow_mo=200` for step-by-step visualization
           - Add print statements at key milestones (e.g., `print("Step X completed")`)
           - Include conditional `time.sleep()` for manual inspection
           - Use `page.pause()` for browser-side debugging
        
        ### Critical Operation Example (Create New Job):
        ```python
        try:
            # Locate and validate button with explicit wait
            create_btn = page.get_by_text("Create New Job", exact=True)
            page.wait_for_selector(create_btn, state="visible", timeout=10000)
            expect(create_btn).to_be_enabled()
            create_btn.click()
            print("Create New Job button clicked")
            
            # Wait for next page element to load
            page.wait_for_selector("label:has-text('Job Name')", state="visible")
        except Exception as e:
            print(f"Error in job creation step: {e}")
            page.screenshot(path="job_creation_error.png")
            time.sleep(15)  # Pause for manual debug
            raise
        ```
        
        ### Example Input:
        "Create a new job, navigate to well design, and add tools robustly"
        
        ### Example Output:
        ```python
        from playwright.sync_api import sync_playwright, expect
        import time

        def run(playwright):
            # Launch browser with debugging options
            browser = playwright.chromium.launch(headless=False, slow_mo=200)
            context = browser.new_context()
            page = context.new_page()
            
            try:
                # Navigate to jobs page with error handling
                page.goto("https://example.com/jobs", wait_until="networkidle")
                print("Navigated to jobs dashboard")
                
                # Step 1: Create New Job
                try:
                    create_btn = page.get_by_text("Create New Job", exact=True)
                    expect(create_btn).to_be_visible()
                    create_btn.click()
                    page.wait_for_selector("label:has-text('Job Name')", state="visible")
                    page.get_by_label("Job Name").fill("test-job-123")
                    page.get_by_text("Create", exact=True).click()
                    page.wait_for_selector("text=Job created", state="visible")
                    print("Job created successfully")
                except Exception as e:
                    print(f"Job creation failed: {e}")
                    page.screenshot(path="job_error.png")
                    raise
                
                # Step 2: Navigate to Well Design (similar error handling)
                try:
                    well_design = page.locator("#nav-panel").get_by_text("Well Design")
                    expect(well_design).to_be_clickable()
                    well_design.click()
                    page.wait_for_load_state("networkidle")
                    print("Navigated to Well Design")
                except Exception as e:
                    print(f"Well Design navigation error: {e}")
                    page.screenshot(path="nav_error.png")
                    raise
                
                # Step 3: Add tools with try-except
                try:
                    page.get_by_text("Add Tool").click()
                    page.get_by_label("Tool Name").fill("drill-bit")
                    page.get_by_text("Save").click()
                    print("Tool added successfully")
                except Exception as e:
                    print(f"Tool addition error: {e}")
                    page.screenshot(path="tool_error.png")
                
            except Exception as e:
                print(f"Test execution error: {e}")
            finally:
                # Pause before closing for manual review
                print("Press Enter to close browser...")
                input()
                context.close()
                browser.close()

        with sync_playwright() as playwright:
            run(playwright)
        ```
        
        INPUT: {text}
        OUTPUT: Robust Playwright automation code with error handling (Python)
        """
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        return LLMChain(llm=llm, prompt=chat_prompt)
    
    def _setup_ui(self):
        """Set up the user interface with clear layouts"""
        # Create title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(
            title_frame, 
            text="Playwright Code Interpreter", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Create main frame
        main_frame = ttk.Notebook(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Workspace tab
        workspace_frame = ttk.Frame(main_frame)
        main_frame.add(workspace_frame, text="Workspace")
        
        # History tab
        history_frame = ttk.Frame(main_frame)
        main_frame.add(history_frame, text="History")
        
        # Input section
        input_frame = ttk.LabelFrame(workspace_frame, text="Input")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Code & natural language input split
        input_splitter = ttk.Frame(input_frame, height=5)
        input_splitter.pack(fill=tk.X, pady=5)
        
        # Code input
        code_frame = ttk.LabelFrame(input_frame, text="Playwright Code Input")
        code_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5), pady=5)
        
        self.code_input = scrolledtext.ScrolledText(
            code_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15, 
            font=("Consolas", 10)
        )
        self.code_input.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Natural language input
        nl_frame = ttk.LabelFrame(input_frame, text="Natural Language Description (robust requirements)")
        nl_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0), pady=5)
        
        self.natural_language = scrolledtext.ScrolledText(
            nl_frame, 
            wrap=tk.WORD, 
            width=40, 
            height=15, 
            font=("Arial", 10)
        )
        self.natural_language.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Button frame
        button_frame = ttk.Frame(workspace_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Function buttons with clear actions
        self.code_to_text_btn = ttk.Button(
            button_frame, 
            text="Code → Natural Language", 
            command=self._convert_code_to_text,
            style="Accent.TButton"
        )
        self.code_to_text_btn.pack(side=tk.LEFT, padx=5)
        
        self.text_to_code_btn = ttk.Button(
            button_frame, 
            text="Natural Language → Robust Code", 
            command=self._convert_text_to_code,
            style="Accent.TButton"
        )
        self.text_to_code_btn.pack(side=tk.LEFT, padx=5)
        
        self.format_code_btn = ttk.Button(
            button_frame, 
            text="Format Code", 
            command=self._format_code,
            style="Accent.TButton"
        )
        self.format_code_btn.pack(side=tk.LEFT, padx=5)
        
        self.execute_btn = ttk.Button(
            button_frame, 
            text="Execute Code", 
            command=self._execute_code,
            style="Accent.TButton"
        )
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_example_btn = ttk.Button(
            button_frame, 
            text="Load Robust Example", 
            command=self._load_example,
            style="Accent.TButton"
        )
        self.load_example_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_all_btn = ttk.Button(
            button_frame, 
            text="Clear All", 
            command=self._clear_all,
            style="Accent.TButton"
        )
        self.clear_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Output sections
        output_frame = ttk.LabelFrame(workspace_frame, text="Playwright Code Output (robust version)")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.code_output = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=10, 
            font=("Consolas", 10)
        )
        self.code_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        result_frame = ttk.LabelFrame(workspace_frame, text="Execution Result")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_output = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=5, 
            font=("Consolas", 10)
        )
        self.result_output.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure button styles
        self.style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def _setup_config_panel(self):
        """Set up LLM and Playwright configuration panel"""
        config_frame = ttk.LabelFrame(self.root, text="LLM & Playwright Configuration")
        config_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # First row: headless mode and timeout
        row1_frame = ttk.Frame(config_frame)
        row1_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(
            row1_frame,
            text="Run Playwright in headless mode",
            variable=self.config["headless"]
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(row1_frame, text="LLM Timeout (seconds):").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1_frame, textvariable=self.config["timeout"], width=5).pack(side=tk.LEFT)
        
        # Second row: model and API configuration
        row2_frame = ttk.Frame(config_frame)
        row2_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(row2_frame, text="Ollama Model:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.config["model_name"], width=30).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(row2_frame, text="Ollama API Base:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row2_frame, textvariable=self.config["api_base"], width=40).pack(side=tk.LEFT)
    
    def _convert_code_to_text(self):
        """Convert Playwright code to natural language instructions"""
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter Playwright code")
            return
        
        try:
            self.status_bar.config(text="Converting code to natural language...")
            self.root.update()
            
            result = self.code_to_text_chain.run(text=code)
            self.natural_language.delete("1.0", tk.END)
            self.natural_language.insert(tk.END, result)
            
            self.status_bar.config(text="Conversion completed")
        except TimeoutError:
            messagebox.showerror("Timeout", "LLM took too long to respond. Please try again.")
            self.status_bar.config(text="Ready")
        except ConnectionError:
            messagebox.showerror("Connection Error", f"Could not connect to Ollama API at {API_BASE}")
            self.status_bar.config(text="Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Conversion error: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _convert_text_to_code(self):
        """Convert natural language to robust Playwright code with error handling"""
        text = self.natural_language.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a natural language description")
            return
        
        try:
            self.status_bar.config(text="Generating robust Playwright code...")
            self.root.update()
            
            result = self.text_to_code_chain.run(text=text)
            cleaned_code = self._cleanse_code_output(result)
            
            # Apply headless mode if enabled
            if self.config["headless"].get():
                cleaned_code = cleaned_code.replace(
                    "launch(headless=False)", 
                    "launch(headless=True)"
                )
            
            self.code_output.delete("1.0", tk.END)
            self.code_output.insert(tk.END, cleaned_code)
            self.status_bar.config(text="Robust code generation completed")
        except TimeoutError:
            messagebox.showerror("Timeout", "LLM took too long. Please try again.")
            self.status_bar.config(text="Ready")
        except ConnectionError:
            messagebox.showerror("Connection Error", f"API error: {self.config['api_base'].get()}")
            self.status_bar.config(text="Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Code generation error: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _cleanse_code_output(self, code):
        """Cleanse LLM-generated code to ensure Python validity"""
        # Remove common LLM response markers
        prefixes = ["```python", "```", "python", "# Code:", "# Output:"]
        for prefix in prefixes:
            if code.startswith(prefix):
                code = code[len(prefix):].lstrip()
        
        if code.endswith("```"):
            code = code[:-3].rstrip()
        
        # Ensure proper function structure
        if "def run(playwright):" not in code:
            code = f"def run(playwright):\n    # Generated code\n{self._indent_code(code)}\n\nwith sync_playwright() as playwright:\n    run(playwright)"
        
        # Add missing imports if needed
        if "from playwright.sync_api import" not in code:
            code = "from playwright.sync_api import sync_playwright, expect\nimport time\n\n" + code
        
        return code
    
    def _indent_code(self, code):
        """Indent code by 4 spaces per line"""
        lines = code.split('\n')
        indented_lines = ['    ' + line for line in lines]
        return '\n'.join(indented_lines)
    
    def _format_code(self):
        """Format the generated code using Python's black formatter"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to format")
            return
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_filename = f.name
            
            # Format the code using black
            try:
                subprocess.run(
                    [sys.executable, '-m', 'black', '-q', temp_filename],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                # If black fails, try autopep8 as fallback
                subprocess.run(
                    [sys.executable, '-m', 'autopep8', '--in-place', temp_filename],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Read the formatted code
            with open(temp_filename, 'r') as f:
                formatted_code = f.read()
            
            # Update the code output
            self.code_output.delete("1.0", tk.END)
            self.code_output.insert(tk.END, formatted_code)
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            self.status_bar.config(text="Code formatted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Formatting error: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _execute_code(self):
        """Execute the generated Playwright code"""
        code = self.code_output.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No code to execute")
            return
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_filename = f.name
            
            self.result_output.delete("1.0", tk.END)
            self.result_output.insert(tk.END, "Executing code...\n")
            self.root.update()
            
            # Execute the code using a subprocess
            process = subprocess.Popen(
                [sys.executable, temp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Capture and display the output
            for line in iter(process.stdout.readline, ''):
                self.result_output.insert(tk.END, line)
                self.result_output.see(tk.END)
                self.root.update()
            
            process.wait()
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            self.status_bar.config(text="Code execution completed")
        except Exception as e:
            messagebox.showerror("Error", f"Execution error: {str(e)}")
            self.status_bar.config(text="Ready")
    
    def _load_example(self):
        """Load a robust example into the input fields"""
        example_text = """
1. Navigate to the login page
2. Enter username and password
3. Click login button
4. Wait for dashboard to load
5. Navigate to user profile
6. Update email address
7. Save changes
8. Verify success message
        """.strip()
        
        self.natural_language.delete("1.0", tk.END)
        self.natural_language.insert(tk.END, example_text)
        
        self.status_bar.config(text="Example loaded - click 'Natural Language → Robust Code' to generate")
    
    def _clear_all(self):
        """Clear all input and output fields"""
        self.code_input.delete("1.0", tk.END)
        self.natural_language.delete("1.0", tk.END)
        self.code_output.delete("1.0", tk.END)
        self.result_output.delete("1.0", tk.END)
        self.status_bar.config(text="All fields cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlaywrightCodeInterpreter(root)
    root.mainloop()
