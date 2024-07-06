import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageTk, ImageFilter
from tkinter import font as tkFont

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        run_optical_flow_script(file_path)

def run_optical_flow_script(file_path):
    try:
        # Correctly define the path to the script in the src folder
        script_path = Path(__file__).parent / 'vel_LK_OF_V1_1.py'
        
        # Print the script path to verify it
        print(f"Script path: {script_path}")
        
        # Check if the script path exists
        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return

        # Print the file path to verify it
        print(f"File path: {file_path}")

        # Check if the file path exists
        if not Path(file_path).exists():
            messagebox.showerror("Error", f"Video file not found: {file_path}")
            return

        # Run the optical flow script with the selected file
        result = subprocess.run([sys.executable, str(script_path), '--input_file', file_path, '--transform', 'True'], capture_output=True, text=True)
        
        # Print the result for debugging
        print(f"Result: {result}")
        
        if result.returncode != 0:
            messagebox.showerror("Error", result.stderr)
        else:
            messagebox.showinfo("Success", "Optical flow calculation completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_enter(e):
    e.widget.config(bg="lightblue", fg="black")

def on_leave(e):
    e.widget.config(bg="SystemButtonFace", fg="black")

def about():
    about_window = tk.Toplevel()
    about_window.title("About")
    about_window.geometry("600x400")

    about_label = tk.Label(about_window, text="About", font=(tkFont.Font(family="Times New Roman", size=22, underline=True)))
    about_label.pack()

    abstract = ("Understanding the dynamics of moving objects is crucial for a variety of computer vision applications, su ch as sports analysis, traffic monitoring and spying. Identifying the exact velocity of the objects caught in video streams is a key component of this understanding. Optical flow, a method based on comparing the intensity variations between successive frames, shows promise as an accurate way to monitor moving objects.")
       
    text_widget = tk.Text(about_window, wrap=tk.WORD, font=("Times New Roman", 16))
    text_widget.pack(expand=1, fill=tk.BOTH, padx=30, pady=20)
    justify_text(text_widget, abstract)

def justify_text(text_widget, text):
    text_widget.delete(1.0, tk.END)  # Clear existing text
    
    lines = text.split('\n')
    for line in lines:
        text_widget.insert(tk.END, line.strip() + '\n')

def run_gui():
    root = tk.Tk()
    root.title("Optical Flow Speed Estimation")

    initial_width = 1500
    initial_height = 800
    root.geometry(f"{initial_width}x{initial_height}")

    header_frame = tk.Frame(root)
    header_frame.pack(fill="x")

    header_label = tk.Label(header_frame, text="Optical Flow Speed Estimation", font=("Garamond", 20))
    header_label.pack(pady=20, padx=50, side="left")

    underline_font = tkFont.Font(family="Garamond", size=18, underline=True)

    about_label = tk.Label(header_frame, text="About", font=underline_font, cursor="hand2")
    about_label.pack(pady=20, padx=30, side="right")
    about_label.bind("<Button-1>", lambda event: about())

    home_label = tk.Label(header_frame, text="Home", font=underline_font, cursor="hand2")
    home_label.pack(pady=20, padx=30, side="right")
    home_label.bind("<Button-1>", lambda e: print("Home Clicked"))

    footer_frame = tk.Frame(root)
    footer_frame.pack(fill="x", side="bottom")

    footer_label = tk.Label(footer_frame, text="all copyrights reserved", font=("Garamond", 12))
    footer_label.pack(anchor="w", padx=30, pady=10)

    image = Image.open("C:/Users/gokul/Desktop/Computer_vision_mini_project/data/cv_image.jpg")
    aspect_ratio = image.width / image.height
    new_height = int(initial_width / aspect_ratio)
    resized_image = image.resize((initial_width, new_height), Image.Resampling.LANCZOS)

    blurred_image = resized_image.filter(ImageFilter.GaussianBlur(1))
    bg_image = ImageTk.PhotoImage(blurred_image)
    
    canvas = tk.Canvas(root, height=initial_height, width=initial_width)
    canvas.pack()

    canvas.create_image(0, 0, anchor="nw", image=bg_image)

    upload_button = tk.Button(root, text="Select Video File", padx=600, pady=5, fg="black", command=select_file)
    upload_button_window = canvas.create_window(100, 100, anchor="nw", window=upload_button)
    
    upload_button.bind("<Enter>", on_enter)
    upload_button.bind("<Leave>", on_leave)

    root.update_idletasks()

    root.mainloop()

if __name__ == "__main__":
    run_gui()
