import customtkinter as ctk
from tkinter import messagebox, filedialog
import os
import re
import cv2
from PIL import Image, ImageTk

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False

# ---------------- Appearance ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# ---------------- Allowed emails ----------------
ALLOWED_EMAILS = {
    "filiptoga@gmail.com",
    "vladcosteasigartau@gmail.com",
    "horeamoldovan07@gmail.com",
    "tatarmihai2@gmail.com"
}

current_user_data = {}
current_frame = None  # track currently displayed frame

# ---------------- Helper functions ----------------
def save_user(name, email, password):
    with open("users.txt", "a", encoding="utf-8") as file:
        file.write(f"{name},{email},{password}\n")

def user_exists(email, password):
    if not os.path.exists("users.txt"):
        return False
    with open("users.txt", "r", encoding="utf-8") as file:
        for line in file:
            stored_name, stored_email, stored_password = line.strip().split(",", 2)
            if email == stored_email and password == stored_password:
                current_user_data.update({
                    'name': stored_name,
                    'email': stored_email,
                    'password': stored_password
                })
                return True
    return False

def is_valid_email(email):
    if EMAIL_VALIDATOR_AVAILABLE:
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))

def email_registered(email):
    if not os.path.exists("users.txt"):
        return False
    with open("users.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1] == email:
                return True
    return False

# ---------------- App ----------------
app = ctk.CTk()
app.title("User App System")
app.geometry("1000x700")
app.update_idletasks()  # ensure correct geometry

# ---------------- UI Constants ----------------
BTN_WIDE = 320
BTN_TALL = 52
TOGGLE_W = 100
TOGGLE_H = 40
BTN_FONT = ctk.CTkFont(size=16, weight="bold")
SMALL_FONT = ctk.CTkFont(size=12)
ENTRY_WIDE = 420
ENTRY_ROW_WIDTH = 360
ENTRY_FONT = ctk.CTkFont(size=16)

# ---------------- Frames ----------------
register_frame = ctk.CTkFrame(app)
login_frame = ctk.CTkFrame(app)
dashboard_frame = ctk.CTkFrame(app)
profile_frame = ctk.CTkFrame(app)
app_frame = ctk.CTkFrame(app)
webcam_frame = ctk.CTkFrame(app)

# ---------------- Frame Helpers ----------------
def reset_entries():
    name_entry.delete(0, 'end')
    name_entry.configure(placeholder_text="Name")
    email_entry.delete(0, 'end')
    email_entry.configure(placeholder_text="Email")
    password_entry.delete(0, 'end')
    password_entry.configure(placeholder_text="Password", show='*')
    login_email_entry.delete(0, 'end')
    login_email_entry.configure(placeholder_text="Email")
    login_password_entry.delete(0, 'end')
    login_password_entry.configure(placeholder_text="Password", show='*')

def slide_frames(from_frame, to_frame, direction="left"):
    app.update_idletasks()
    width = app.winfo_width()
    step = 20
    delay = 0.003

    if direction == "left":
        start_to = width
        delta = -step
    else:
        start_to = -width
        delta = step

    # Place frames at proper starting positions
    from_frame.place(x=0, y=0, relwidth=1, relheight=1)
    to_frame.place(x=start_to, y=0, relwidth=1, relheight=1)
    app.update()

    for _ in range(width // step):
        from_frame.place_configure(x=from_frame.winfo_x() + delta)
        to_frame.place_configure(x=to_frame.winfo_x() + delta)
        app.update()
        time.sleep(delay)

    from_frame.place_forget()
    to_frame.place(x=0, y=0, relwidth=1, relheight=1)

def show_frame(frame):
    global current_frame
    if frame == register_frame:
        reset_entries()
        name_entry.focus()
    elif frame == login_frame:
        reset_entries()
        login_email_entry.focus()
    elif frame == webcam_frame:
        start_webcam()

    if current_frame and current_frame != frame:
        # Stop webcam if leaving webcam frame
        if current_frame == webcam_frame:
            stop_webcam()
        
        forward_frames = [register_frame, login_frame, dashboard_frame, profile_frame, app_frame, webcam_frame]
        if forward_frames.index(frame) > forward_frames.index(current_frame):
            slide_frames(current_frame, frame, direction="left")
        else:
            slide_frames(current_frame, frame, direction="right")
    else:
        frame.place(x=0, y=0, relwidth=1, relheight=1)

    current_frame = frame  # update current frame

# ---------------- Registration ----------------
ctk.CTkLabel(register_frame, text="Create Your Account", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=30)
name_entry = ctk.CTkEntry(register_frame, width=ENTRY_WIDE, font=ENTRY_FONT, placeholder_text="Name")
name_entry.pack(pady=8)
email_entry = ctk.CTkEntry(register_frame, width=ENTRY_WIDE, font=ENTRY_FONT, placeholder_text="Email")
email_entry.pack(pady=8)
password_row = ctk.CTkFrame(register_frame)
password_row.pack(pady=8)
password_entry = ctk.CTkEntry(password_row, width=ENTRY_ROW_WIDTH, font=ENTRY_FONT, show='*', placeholder_text="Password")
password_entry.pack(side='left')

_reg_pwd_shown = False
def toggle_reg_pwd():
    global _reg_pwd_shown
    _reg_pwd_shown = not _reg_pwd_shown
    password_entry.configure(show='' if _reg_pwd_shown else '*')
    reg_toggle.configure(text='Hide' if _reg_pwd_shown else 'Show')

reg_toggle = ctk.CTkButton(password_row, text='Show', width=TOGGLE_W, height=TOGGLE_H, font=SMALL_FONT, command=toggle_reg_pwd)
reg_toggle.pack(side='left', padx=(8, 0))

def register():
    name = name_entry.get()
    email = email_entry.get()
    password = password_entry.get()
    if not name or not email or not password:
        messagebox.showwarning("Input Error", "Please fill all fields.")
        return
    if not is_valid_email(email):
        messagebox.showwarning("Input Error", "Please enter a valid email address.")
        return
    if email not in ALLOWED_EMAILS:
        messagebox.showerror("Registration Failed", "Email not in the database.")
        return
    if email_registered(email):
        messagebox.showerror("Registration Failed", "Email already registered.")
        return
    save_user(name, email, password)
    messagebox.showinfo("Success", "Registration successful!")
    show_frame(login_frame)

ctk.CTkButton(register_frame, text="Register", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=register).pack(pady=20)
ctk.CTkButton(register_frame, text="Already have an account? Login", width=BTN_WIDE, height=36, font=SMALL_FONT, fg_color="transparent",
              command=lambda: show_frame(login_frame)).pack()

# ---------------- Login ----------------
ctk.CTkLabel(login_frame, text="Login to Your Account", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=30)
login_email_entry = ctk.CTkEntry(login_frame, width=ENTRY_WIDE, font=ENTRY_FONT, placeholder_text="Email")
login_email_entry.pack(pady=8)
login_pwd_row = ctk.CTkFrame(login_frame)
login_pwd_row.pack(pady=8)
login_password_entry = ctk.CTkEntry(login_pwd_row, width=ENTRY_ROW_WIDTH, font=ENTRY_FONT, show='*', placeholder_text="Password")
login_password_entry.pack(side='left')

_login_pwd_shown = False
def toggle_login_pwd():
    global _login_pwd_shown
    _login_pwd_shown = not _login_pwd_shown
    login_password_entry.configure(show='' if _login_pwd_shown else '*')
    login_toggle.configure(text='Hide' if _login_pwd_shown else 'Show')

login_toggle = ctk.CTkButton(login_pwd_row, text='Show', width=TOGGLE_W, height=TOGGLE_H, font=SMALL_FONT, command=toggle_login_pwd)
login_toggle.pack(side='left', padx=(8, 0))

def login():
    email = login_email_entry.get()
    password = login_password_entry.get()
    if user_exists(email, password):
        if current_user_data.get('email') not in ALLOWED_EMAILS:
            messagebox.showerror("Login Failed", "Email not in the database.")
            current_user_data.clear()
            return
        dashboard_label.configure(text=f"Welcome, {current_user_data.get('name', '')}!")
        show_frame(dashboard_frame)
    else:
        messagebox.showerror("Login Failed", "Invalid email or password.")

ctk.CTkButton(login_frame, text="Login", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=login).pack(pady=16)
ctk.CTkButton(login_frame, text="No account? Register", width=BTN_WIDE, height=36, font=SMALL_FONT, fg_color="transparent",
              command=lambda: show_frame(register_frame)).pack()

# ---------------- Dashboard ----------------
dashboard_label = ctk.CTkLabel(dashboard_frame, text="", font=ctk.CTkFont(size=20))
dashboard_label.pack(pady=30)

ctk.CTkButton(dashboard_frame, text="Profile", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: (update_profile(), show_frame(profile_frame))).pack(pady=8)
ctk.CTkButton(dashboard_frame, text="Choose a file", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: show_frame(app_frame)).pack(pady=8)
ctk.CTkButton(dashboard_frame, text="Webcam", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: show_frame(webcam_frame)).pack(pady=8)
ctk.CTkButton(dashboard_frame, text="Logout", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: (current_user_data.clear(), show_frame(login_frame))).pack(pady=20)

# ---------------- Profile ----------------
ctk.CTkLabel(profile_frame, text="Your Profile", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
profile_username = ctk.CTkLabel(profile_frame, text="", font=ctk.CTkFont(size=16))
profile_username.pack(pady=8)
profile_email = ctk.CTkLabel(profile_frame, text="", font=ctk.CTkFont(size=16))
profile_email.pack(pady=8)
profile_password = ctk.CTkLabel(profile_frame, text="", font=ctk.CTkFont(size=16))
profile_password.pack(pady=8)

_profile_show_password = False
def toggle_show_password():
    global _profile_show_password
    _profile_show_password = not _profile_show_password
    if _profile_show_password:
        profile_password.configure(text=f"Password: {current_user_data.get('password', '')}")
        show_pwd_btn.configure(text='Hide password')
    else:
        profile_password.configure(text=f"Password: {'*' * len(current_user_data.get('password', ''))}")
        show_pwd_btn.configure(text='Show password')

show_pwd_btn = ctk.CTkButton(profile_frame, text="Show password", width=220, height=44, font=SMALL_FONT, command=toggle_show_password)
show_pwd_btn.pack(pady=(0, 12))
ctk.CTkButton(profile_frame, text="Back to Dashboard", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: show_frame(dashboard_frame)).pack()

# ---------------- App Selector ----------------
ctk.CTkLabel(app_frame, text="Choose an App", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
selected_media_path = None
selected_media_type = None

def choose_video():
    global selected_media_path, selected_media_type
    selected_media_type = 'video'
    path = filedialog.askopenfilename(title="Select a video file", filetypes=[("MP4 files", "*.mp4"), ("All files", "*")])
    if path:
        selected_media_path = path
        messagebox.showinfo("File Selected", f"Selected video:\n{selected_media_path}")

def choose_photo():
    global selected_media_path, selected_media_type
    selected_media_type = 'photo'
    path = filedialog.askopenfilename(title="Select a photo file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif"), ("All files", "*")])
    if path:
        selected_media_path = path
        messagebox.showinfo("File Selected", f"Selected photo:\n{selected_media_path}")

ctk.CTkButton(app_frame, text="Video", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=choose_video).pack(pady=8)
ctk.CTkButton(app_frame, text="Photo", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=choose_photo).pack(pady=8)
ctk.CTkButton(app_frame, text="Back to Dashboard", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, command=lambda: show_frame(dashboard_frame)).pack(pady=20)

# ---------------- Webcam ----------------
ctk.CTkLabel(webcam_frame, text="Webcam Feed", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

webcam_label = ctk.CTkLabel(webcam_frame, text="")
webcam_label.pack(pady=10)

webcam_active = False
webcam_capture = None

def update_webcam_feed():
    global webcam_active, webcam_capture
    if webcam_active and webcam_capture is not None:
        ret, frame = webcam_capture.read()
        if ret:
            # Resize frame to fit in the window
            frame = cv2.resize(frame, (640, 480))
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            # Convert to CTkImage
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            webcam_label.configure(image=ctk_img)
            webcam_label.image = ctk_img
        # Schedule next update
        webcam_label.after(10, update_webcam_feed)

def start_webcam():
    global webcam_active, webcam_capture
    if not webcam_active:
        webcam_capture = cv2.VideoCapture(0)
        if webcam_capture.isOpened():
            webcam_active = True
            update_webcam_feed()
        else:
            messagebox.showerror("Webcam Error", "Could not access webcam.")

def stop_webcam():
    global webcam_active, webcam_capture
    webcam_active = False
    if webcam_capture is not None:
        webcam_capture.release()
        webcam_capture = None
    webcam_label.configure(image=None, text="Webcam stopped")

ctk.CTkButton(webcam_frame, text="Back to Dashboard", width=BTN_WIDE, height=BTN_TALL, font=BTN_FONT, 
              command=lambda: (stop_webcam(), show_frame(dashboard_frame))).pack(pady=20)

# ---------------- Profile Update ----------------
def update_profile():
    profile_username.configure(text=f"Name: {current_user_data.get('name', '')}")
    profile_email.configure(text=f"Email: {current_user_data.get('email', '')}")
    profile_password.configure(text=f"Password: {'*' * len(current_user_data.get('password', ''))}")

# ---------------- Start App ----------------
show_frame(register_frame)
app.mainloop()