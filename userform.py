import tkinter as tk
from tkinter import messagebox
import os

# Store current logged-in user data
current_user_data = {}

# ---------------------------
# Helper functions
# ---------------------------

def save_user(username, email, password):
    with open("users.txt", "a") as file:
        file.write(f"{username},{email},{password}\n")

def user_exists(username, password):
    if not os.path.exists("users.txt"):
        return False

    with open("users.txt", "r") as file:
        for line in file:
            stored_username, stored_email, stored_password = line.strip().split(",")
            if username == stored_username and password == stored_password:
                current_user_data['username'] = stored_username
                current_user_data['email'] = stored_email
                current_user_data['password'] = stored_password
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

def show_dashboard(username):
    hide_all_frames()
    dashboard_label.config(text=f"Welcome, {username}!")
    dashboard_frame.pack(fill=tk.BOTH, expand=True)

def logout():
    hide_all_frames()
    current_user_data.clear()
    show_login()

def show_profile():
    hide_all_frames()
    profile_username.config(text=f"Username: {current_user_data.get('username', '')}")
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
    username = username_entry.get()
    email = email_entry.get()
    password = password_entry.get()

    if not username or not email or not password:
        messagebox.showwarning("Input Error", "Please fill all fields.")
        return

    save_user(username, email, password)
    messagebox.showinfo("Success", "Registration successful!")
    show_login()

def login():
    username = login_username_entry.get()
    password = login_password_entry.get()

    if user_exists(username, password):
        messagebox.showinfo("Login Success", f"Welcome, {username}!")
        show_dashboard(username)
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

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

tk.Label(inner_form, text="Username:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=1, column=0, sticky="e", padx=10, pady=10)
username_entry = tk.Entry(inner_form, font=("Arial", 16), width=25)
username_entry.grid(row=1, column=1, padx=10, pady=10)

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

tk.Label(inner_login, text="Username:", font=("Arial", 16), bg="#2e8b57", fg="white").grid(row=1, column=0, sticky="e", padx=10, pady=10)
login_username_entry = tk.Entry(inner_login, font=("Arial", 16), width=25)
login_username_entry.grid(row=1, column=1, padx=10, pady=10)

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

tk.Button(profile_frame, text="Back to Dashboard", font=("Arial", 14), command=lambda: show_dashboard(current_user_data.get('username', ''))).pack(pady=40)

# ---------------------------
# App Selector Frame
# ---------------------------

app_frame = tk.Frame(root, padx=20, pady=20, bg="lightgrey")

tk.Label(app_frame, text="Choose an App", font=("Arial", 24, "bold"), bg="lightgrey").pack(pady=30)

tk.Button(app_frame, text="Calculator (coming soon)", width=25, font=("Arial", 14)).pack(pady=10)
tk.Button(app_frame, text="Notes (coming soon)", width=25, font=("Arial", 14)).pack(pady=10)
tk.Button(app_frame, text="Back to Dashboard", width=25, font=("Arial", 14), command=lambda: show_dashboard(current_user_data.get('username', ''))).pack(pady=40)

# ---------------------------
# Start with Registration Screen
# ---------------------------

show_registration()
root.mainloop()

