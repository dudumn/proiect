import tkinter as tk
from tkinter import messagebox
import os
try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False
import re

# Allowed emails (start with filiptoga@gmail.com)
ALLOWED_EMAILS = {"filiptoga@gmail.com", "vladcosteasigartau@gmail.com"}

# Store current logged-in user data
current_user_data = {}

# ---------------------------
# Helper functions
# ---------------------------

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
                current_user_data['name'] = stored_name
                current_user_data['email'] = stored_email
                current_user_data['password'] = stored_password
                return True
    return False

def is_valid_email(email: str) -> bool:
    if EMAIL_VALIDATOR_AVAILABLE:
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    # Basic regex fallback
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

def email_registered(email: str) -> bool:
    if not os.path.exists("users.txt"):
        return False
    with open("users.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[1] == email:
                return True
    return False

# ---------------------------
# Screen switching
# ---------------------------

def hide_all_frames():
    for frame in [form_frame, login_frame, dashboard_frame, profile_frame, app_frame]:
        frame.pack_forget()

def show_registration():
    hide_all_frames()
    form_frame.pack(fill=tk.BOTH, expand=True)

def show_login():
    hide_all_frames()
    login_frame.pack(fill=tk.BOTH, expand=True)

def show_dashboard(name):
    hide_all_frames()
    dashboard_label.config(text=f"Welcome, {name}!")
    dashboard_frame.pack(fill=tk.BOTH, expand=True)

def logout():
    hide_all_frames()
    current_user_data.clear()
    show_login()

def show_profile():
    hide_all_frames()
    profile_username.config(text=f"Name: {current_user_data.get('name', '')}")
    profile_email.config(text=f"Email: {current_user_data.get('email', '')}")
    profile_password.config(text=f"Password: {'*' * len(current_user_data.get('password', ''))}")
    profile_frame.pack(fill=tk.BOTH, expand=True)

def show_app_selector():
    hide_all_frames()
    app_frame.pack(fill=tk.BOTH, expand=True)

# ---------------------------
# Functional logic
# ---------------------------

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
        messagebox.showerror("Registration Failed", "Email not in the database")
        return

    if email_registered(email):
        messagebox.showerror("Registration Failed", "E--mail already regisstered.")
        return

    save_user(name, email, password)
    messagebox.showinfo("Success", "Registration successful!")
    show_login()

def login():
    email = login_email_entry.get()
    password = login_password_entry.get()

    if user_exists(email, password):
        if current_user_data.get('email') not in ALLOWED_EMAILS:
            messagebox.showerror("Login Failed", "Email not in the database")
            current_user_data.clear()
            return

        messagebox.showinfo("Login Success", f"Welcome, {current_user_data.get('name', '')}!")
        show_dashboard(current_user_data.get('name', ''))
    else:
        messagebox.showerror("Login Failed", "Invalid email or password")

# ---------------------------
# Main Window Setup
# ---------------------------

root = tk.Tk()
root.title("User App System")
root.configure(background="#2e8b57")

# Make fullscreen and allow exiting with ESC
root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

# ---------------------------
# Registration Frame (Fullscreen & Centered)
# ---------------------------

form_frame = tk.Frame(root, bg="#2e8b57")

inner_form = tk.Frame(form_frame, bg="#2e8b57")
inner_form.pack(expand=True)

tk.Label(inner_form, text="Create Your Account", font=("Arial", 28, "bold"), fg="white", bg="#2e8b57").grid(row=0, column=0, columnspan=2, pady=(0, 40))

tk.Label(inner_form, text="Name:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=1, column=0, sticky="e", padx=10, pady=10)
name_entry = tk.Entry(inner_form, font=("Arial", 16), width=25)
name_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(inner_form, text="Email:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=2, column=0, sticky="e", padx=10, pady=10)
email_entry = tk.Entry(inner_form, font=("Arial", 16), width=25)
email_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Label(inner_form, text="Password:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=3, column=0, sticky="e", padx=10, pady=10)
password_entry = tk.Entry(inner_form, show="*", font=("Arial", 16), width=25)
password_entry.grid(row=3, column=1, padx=10, pady=10)

tk.Button(inner_form, text="Register", font=("Arial", 16, "bold"), bg="#4CAF50", fg="white", width=20, command=register).grid(row=4, column=0, columnspan=2, pady=(30, 10))

tk.Button(inner_form, text="Already have an account? Login", font=("Arial", 12), bg="#2e8b57", fg="white", bd=0, activebackground="#2e8b57", command=show_login).grid(row=5, column=0, columnspan=2, pady=(10, 0))

# ---------------------------
# Login Frame (Fullscreen & Centered)
# ---------------------------

login_frame = tk.Frame(root, bg="#2e8b57")

inner_login = tk.Frame(login_frame, bg="#2e8b57")
inner_login.pack(expand=True)

tk.Label(inner_login, text="Login to Your Account", font=("Arial", 28, "bold"), fg="white", bg="#2e8b57").grid(row=0, column=0, columnspan=2, pady=(0, 40))

tk.Label(inner_login, text="Email:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=1, column=0, sticky="e", padx=10, pady=10)
login_email_entry = tk.Entry(inner_login, font=("Arial", 16), width=25)
login_email_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(inner_login, text="Password:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=2, column=0, sticky="e", padx=10, pady=10)
login_password_entry = tk.Entry(inner_login, show="*", font=("Arial", 16), width=25)
login_password_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Button(inner_login, text="Login", font=("Arial", 16, "bold"), bg="#4CAF50", fg="white", width=20, command=login).grid(row=3, column=0, columnspan=2, pady=(30, 10))

tk.Button(inner_login, text="No account? Register", font=("Arial", 12), bg="#2e8b57", fg="white", bd=0, activebackground="#2e8b57", command=show_registration).grid(row=4, column=0, columnspan=2, pady=(10, 0))

# ---------------------------
# Dashboard Frame
# ---------------------------

dashboard_frame = tk.Frame(root, padx=20, pady=20, bg="white")

dashboard_label = tk.Label(dashboard_frame, text="", font=("Helvetica", 24, "bold"), bg="white", fg="#333")
dashboard_label.pack(pady=40)

tk.Button(dashboard_frame, text="Profile", width=25, font=("Arial", 14), command=show_profile).pack(pady=10)
tk.Button(dashboard_frame, text="App Selector", width=25, font=("Arial", 14), command=show_app_selector).pack(pady=10)
tk.Button(dashboard_frame, text="Logout", width=25, font=("Arial", 14), command=logout).pack(pady=40)

# ---------------------------
# Profile Frame
# ---------------------------

profile_frame = tk.Frame(root, padx=20, pady=20, bg="lightblue")

tk.Label(profile_frame, text="Your Profile", font=("Arial", 24, "bold"), bg="lightblue").pack(pady=30)

profile_username = tk.Label(profile_frame, text="", font=("Arial", 16), bg="lightblue")
profile_username.pack(pady=10)

profile_email = tk.Label(profile_frame, text="", font=("Arial", 16), bg="lightblue")
profile_email.pack(pady=10)

profile_password = tk.Label(profile_frame, text="", font=("Arial", 16), bg="lightblue")
profile_password.pack(pady=10)

tk.Button(profile_frame, text="Back to Dashboard", font=("Arial", 14), command=lambda: show_dashboard(current_user_data.get('name', ''))).pack(pady=40)

# ---------------------------
# App Selector Frame
# ---------------------------

app_frame = tk.Frame(root, padx=20, pady=20, bg="lightgrey")

tk.Label(app_frame, text="Choose an App", font=("Arial", 24, "bold"), bg="lightgrey").pack(pady=30)

tk.Button(app_frame, text="Calculator (coming soon)", width=25, font=("Arial", 14)).pack(pady=10)
tk.Button(app_frame, text="Notes (coming soon)", width=25, font=("Arial", 14)).pack(pady=10)
tk.Button(app_frame, text="Back to Dashboard", width=25, font=("Arial", 14), command=lambda: show_dashboard(current_user_data.get('name', ''))).pack(pady=40)

# ---------------------------
# Start with Registration Screen
# ---------------------------

show_registration()
root.mainloop()