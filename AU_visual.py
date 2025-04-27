import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

class FACSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("FACS Facial Expression Simulator")
        self.root.geometry("1000x700")
        
        # Define neutral face landmarks (68 points)
        self.neutral_face = np.zeros((68, 2))
        
        # Jaw points (0-16) - not shown as per requirements
        # We initialize them but won't display them
        
        # Eyebrows (17-26)
        # Right eyebrow
        self.neutral_face[17] = [155, 100]
        self.neutral_face[18] = [145, 95]
        self.neutral_face[19] = [135, 95]
        self.neutral_face[20] = [125, 95]
        self.neutral_face[21] = [115, 100]
        # Left eyebrow
        self.neutral_face[22] = [185, 100]
        self.neutral_face[23] = [195, 95]
        self.neutral_face[24] = [205, 95]
        self.neutral_face[25] = [215, 95]
        self.neutral_face[26] = [225, 100]
        
        # Nose (27-35)
        self.neutral_face[27] = [150, 120]
        self.neutral_face[28] = [150, 130]
        self.neutral_face[29] = [150, 140]
        self.neutral_face[30] = [145, 145]
        self.neutral_face[31] = [150, 145]
        self.neutral_face[32] = [155, 145]
        self.neutral_face[33] = [140, 140]
        self.neutral_face[34] = [150, 140]
        self.neutral_face[35] = [160, 140]
        
        # Eyes (36-47)
        # Right eye
        self.neutral_face[36] = [135, 115]
        self.neutral_face[37] = [140, 112]
        self.neutral_face[38] = [145, 112]
        self.neutral_face[39] = [150, 115]
        self.neutral_face[40] = [145, 118]
        self.neutral_face[41] = [140, 118]
        # Left eye
        self.neutral_face[42] = [190, 115]
        self.neutral_face[43] = [195, 112]
        self.neutral_face[44] = [200, 112]
        self.neutral_face[45] = [205, 115]
        self.neutral_face[46] = [200, 118]
        self.neutral_face[47] = [195, 118]
        
        # Mouth (48-67)
        self.neutral_face[48] = [130, 170]
        self.neutral_face[49] = [140, 165]
        self.neutral_face[50] = [145, 165]
        self.neutral_face[51] = [150, 165]
        self.neutral_face[52] = [155, 165]
        self.neutral_face[53] = [160, 165]
        self.neutral_face[54] = [170, 170]
        self.neutral_face[55] = [160, 175]
        self.neutral_face[56] = [155, 176]
        self.neutral_face[57] = [150, 176]
        self.neutral_face[58] = [145, 176]
        self.neutral_face[59] = [140, 175]
        self.neutral_face[60] = [135, 170]
        self.neutral_face[61] = [145, 170]
        self.neutral_face[62] = [150, 170]
        self.neutral_face[63] = [155, 170]
        self.neutral_face[64] = [165, 170]
        self.neutral_face[65] = [155, 170]
        self.neutral_face[66] = [150, 170]
        self.neutral_face[67] = [145, 170]
        
        # Initialize the current face points
        self.face_points = np.copy(self.neutral_face)
        
        # Initialize action units dictionary (AU name: value 0-100)
        self.action_units = {
            'AU1': 0,  # Inner Brow Raiser
            'AU2': 0,  # Outer Brow Raiser
            'AU4': 0,  # Brow Lowerer
            'AU5': 0,  # Upper Lid Raiser
            'AU6': 0,  # Cheek Raiser
            'AU7': 0,  # Lid Tightener
            'AU9': 0,  # Nose Wrinkler
            'AU10': 0, # Upper Lip Raiser
            'AU12': 0, # Lip Corner Puller (smile)
            'AU15': 0, # Lip Corner Depressor
            'AU17': 0, # Chin Raiser
            'AU20': 0, # Lip Stretcher
            'AU23': 0, # Lip Tightener
            'AU25': 0, # Lips Part
            'AU26': 0, # Jaw Drop
            'AU43': 0  # Eyes Closed
        }
        
        # Predefined expressions
        self.expressions = {
            'happy': {
                'AU6': 60,
                'AU12': 80,
                'AU25': 30
            },
            'sad': {
                'AU1': 50,
                'AU4': 30,
                'AU15': 70
            },
            'angry': {
                'AU4': 80,
                'AU5': 40,
                'AU7': 50,
                'AU23': 60
            },
            'surprised': {
                'AU1': 70,
                'AU2': 70,
                'AU5': 80,
                'AU26': 70
            },
            'disgusted': {
                'AU9': 80,
                'AU10': 60,
                'AU15': 50,
                'AU17': 40
            },
            'fearful': {
                'AU1': 60,
                'AU2': 60,
                'AU4': 30,
                'AU5': 70,
                'AU20': 40,
                'AU26': 30
            }
        }
        
        # Create the main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for the face visualization
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create the figure and canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create button frame
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Add buttons for preset expressions
        ttk.Button(button_frame, text="Reset", command=self.reset_face).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Happy", command=lambda: self.set_expression('happy')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Sad", command=lambda: self.set_expression('sad')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Angry", command=lambda: self.set_expression('angry')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Surprised", command=lambda: self.set_expression('surprised')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Disgusted", command=lambda: self.set_expression('disgusted')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Fearful", command=lambda: self.set_expression('fearful')).pack(side=tk.LEFT, padx=5)
        
        # Create right frame for AU controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Add title for AU controls
        ttk.Label(right_frame, text="Facial Action Units", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create scrollable frame for sliders
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize value_labels dictionary before creating sliders
        self.sliders = {}
        self.value_labels = {}
        
        # Create AU sliders
        row = 0
        for au, value in self.action_units.items():
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = ttk.Label(frame, text=f"{au}:", width=10)
            label.pack(side=tk.LEFT)
            
            slider = ttk.Scale(
                frame,
                from_=0,
                to=100,
                orient="horizontal",
                value=value,
                command=lambda val, au=au: self.update_au(au, val)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.sliders[au] = slider
            
            value_label = ttk.Label(frame, text=str(value), width=5)
            value_label.pack(side=tk.LEFT)
            self.value_labels[au] = value_label
            
            row += 1
        
        # Add AU descriptions
        desc_frame = ttk.LabelFrame(right_frame, text="Action Unit Descriptions")
        desc_frame.pack(fill=tk.X, pady=10, padx=5)
        
        descriptions = [
            ("AU1", "Inner Brow Raiser"),
            ("AU2", "Outer Brow Raiser"),
            ("AU4", "Brow Lowerer"),
            ("AU5", "Upper Lid Raiser"),
            ("AU6", "Cheek Raiser"),
            ("AU7", "Lid Tightener"),
            ("AU9", "Nose Wrinkler"),
            ("AU10", "Upper Lip Raiser"),
            ("AU12", "Lip Corner Puller (smile)"),
            ("AU15", "Lip Corner Depressor"),
            ("AU17", "Chin Raiser"),
            ("AU20", "Lip Stretcher"),
            ("AU23", "Lip Tightener"),
            ("AU25", "Lips Part"),
            ("AU26", "Jaw Drop"),
            ("AU43", "Eyes Closed")
        ]
        
        for i, (au, desc) in enumerate(descriptions):
            ttk.Label(desc_frame, text=f"{au}: {desc}").pack(anchor="w", padx=5)
        
        # Draw the initial face
        self.update_face()
    
    def update_au(self, au, value):
        # Update the action unit value and related label
        try:
            val = float(value)
            self.action_units[au] = val
            if au in self.value_labels:
                self.value_labels[au].config(text=f"{int(val)}")
            self.update_face()
        except Exception as e:
            print(f"Error updating AU {au}: {e}")
    
    def apply_action_units(self):
        # Start with neutral face
        new_points = np.copy(self.neutral_face)
        
        # AU1: Inner Brow Raiser
        if self.action_units['AU1'] > 0:
            intensity = self.action_units['AU1'] / 100
            # Affect inner brow points (18-21)
            for i in range(18, 22):
                new_points[i, 1] = self.neutral_face[i, 1] - 15 * intensity
        
        # AU2: Outer Brow Raiser
        if self.action_units['AU2'] > 0:
            intensity = self.action_units['AU2'] / 100
            # Affect outer brow points (17, 22-26)
            new_points[17, 1] = self.neutral_face[17, 1] - 10 * intensity
            for i in range(22, 27):
                new_points[i, 1] = self.neutral_face[i, 1] - 10 * intensity
        
        # AU4: Brow Lowerer
        if self.action_units['AU4'] > 0:
            intensity = self.action_units['AU4'] / 100
            # Lower brows and bring them together
            for i in range(17, 22):
                new_points[i, 0] = self.neutral_face[i, 0] + 3 * intensity
                new_points[i, 1] = self.neutral_face[i, 1] + 8 * intensity
            for i in range(22, 27):
                new_points[i, 0] = self.neutral_face[i, 0] - 3 * intensity
                new_points[i, 1] = self.neutral_face[i, 1] + 8 * intensity
        
        # AU5: Upper Lid Raiser
        if self.action_units['AU5'] > 0:
            intensity = self.action_units['AU5'] / 100
            # Widen eyes (upper lids)
            for i in [37, 38, 43, 44]:
                new_points[i, 1] = self.neutral_face[i, 1] - 5 * intensity
        
        # AU6: Cheek Raiser
        if self.action_units['AU6'] > 0:
            intensity = self.action_units['AU6'] / 100
            # Raise cheeks, affect lower eye points
            for i in [40, 41, 46, 47]:
                new_points[i, 1] = self.neutral_face[i, 1] + 3 * intensity
        
        # AU7: Lid Tightener
        if self.action_units['AU7'] > 0:
            intensity = self.action_units['AU7'] / 100
            # Tighten eyelids
            for i in [37, 38, 43, 44]:
                new_points[i, 1] = self.neutral_face[i, 1] + 3 * intensity
            for i in [40, 41, 46, 47]:
                new_points[i, 1] = self.neutral_face[i, 1] - 3 * intensity
        
        # AU9: Nose Wrinkler
        if self.action_units['AU9'] > 0:
            intensity = self.action_units['AU9'] / 100
            # Wrinkle the nose
            for i in range(27, 31):
                new_points[i, 1] = self.neutral_face[i, 1] - 4 * intensity
            for i in range(31, 36):
                new_points[i, 1] = self.neutral_face[i, 1] + 3 * intensity
        
        # AU10: Upper Lip Raiser
        if self.action_units['AU10'] > 0:
            intensity = self.action_units['AU10'] / 100
            # Raise upper lip
            for i in range(48, 55):
                new_points[i, 1] = self.neutral_face[i, 1] - 6 * intensity
        
        # AU12: Lip Corner Puller (smile)
        if self.action_units['AU12'] > 0:
            intensity = self.action_units['AU12'] / 100
            # Pull lip corners up and out
            new_points[48, 0] = self.neutral_face[48, 0] - 8 * intensity
            new_points[48, 1] = self.neutral_face[48, 1] - 5 * intensity
            new_points[54, 0] = self.neutral_face[54, 0] + 8 * intensity
            new_points[54, 1] = self.neutral_face[54, 1] - 5 * intensity
            
            # Adjust surrounding mouth points
            new_points[49, 0] = self.neutral_face[49, 0] - 4 * intensity
            new_points[49, 1] = self.neutral_face[49, 1] - 2 * intensity
            new_points[53, 0] = self.neutral_face[53, 0] + 4 * intensity
            new_points[53, 1] = self.neutral_face[53, 1] - 2 * intensity
            
            for i in [58, 59, 60]:
                new_points[i, 1] = self.neutral_face[i, 1] - 1 * intensity
        
        # AU15: Lip Corner Depressor
        if self.action_units['AU15'] > 0:
            intensity = self.action_units['AU15'] / 100
            # Pull lip corners down
            new_points[48, 1] = self.neutral_face[48, 1] + 8 * intensity
            new_points[54, 1] = self.neutral_face[54, 1] + 8 * intensity
            # Adjust surrounding mouth points
            new_points[57, 1] = self.neutral_face[57, 1] + 3 * intensity
            new_points[55, 1] = self.neutral_face[55, 1] + 3 * intensity
        
        # AU17: Chin Raiser
        if self.action_units['AU17'] > 0:
            intensity = self.action_units['AU17'] / 100
            # Raise chin, affect lower lip
            for i in range(57, 68):
                new_points[i, 1] = self.neutral_face[i, 1] - 5 * intensity
        
        # AU20: Lip Stretcher
        if self.action_units['AU20'] > 0:
            intensity = self.action_units['AU20'] / 100
            # Stretch lips horizontally
            new_points[48, 0] = self.neutral_face[48, 0] - 8 * intensity
            new_points[54, 0] = self.neutral_face[54, 0] + 8 * intensity
            
            # Adjust surrounding mouth points
            new_points[49, 0] = self.neutral_face[49, 0] - 4 * intensity
            new_points[53, 0] = self.neutral_face[53, 0] + 4 * intensity
            new_points[55, 0] = self.neutral_face[55, 0] + 4 * intensity
            new_points[59, 0] = self.neutral_face[59, 0] - 4 * intensity
        
        # AU23: Lip Tightener
        if self.action_units['AU23'] > 0:
            intensity = self.action_units['AU23'] / 100
            # Tighten lips
            for i in range(48, 55):
                new_points[i, 1] = self.neutral_face[i, 1] + 2 * intensity
            for i in range(55, 60):
                new_points[i, 1] = self.neutral_face[i, 1] - 2 * intensity
        
        # AU25: Lips Part
        if self.action_units['AU25'] > 0:
            intensity = self.action_units['AU25'] / 100
            # Part lips
            for i in range(50, 53):
                new_points[i, 1] = self.neutral_face[i, 1] - 5 * intensity
            for i in range(56, 59):
                new_points[i, 1] = self.neutral_face[i, 1] + 5 * intensity
        
        # AU26: Jaw Drop
        if self.action_units['AU26'] > 0:
            intensity = self.action_units['AU26'] / 100
            # Drop jaw, open mouth wider
            for i in range(56, 68):
                new_points[i, 1] = self.neutral_face[i, 1] + 15 * intensity
        
        # AU43: Eyes Closed
        if self.action_units['AU43'] > 0:
            intensity = self.action_units['AU43'] / 100
            # Close eyes by moving upper and lower lids together
            for i in [37, 38, 43, 44]:
                new_points[i, 1] = self.neutral_face[i, 1] + 4 * intensity
            for i in [40, 41, 46, 47]:
                new_points[i, 1] = self.neutral_face[i, 1] - 4 * intensity
        
        self.face_points = new_points
    
    def update_face(self):
        # Apply action units to get updated face points
        self.apply_action_units()
        
        # Clear the current plot
        self.ax.clear()
        
        # Set the limits and aspect
        self.ax.set_xlim(80, 260)
        self.ax.set_ylim(210, 70)  # Reversed y-axis to fix upside-down issue
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw face points (skipping jaw points 0-16)
        self.ax.scatter(self.face_points[17:, 0], self.face_points[17:, 1], color='black', s=10)
        
        # Draw eyebrows
        self.ax.plot(self.face_points[17:22, 0], self.face_points[17:22, 1], 'k-', linewidth=2)
        self.ax.plot(self.face_points[22:27, 0], self.face_points[22:27, 1], 'k-', linewidth=2)
        
        # Draw nose
        self.ax.plot(self.face_points[27:36, 0], self.face_points[27:36, 1], 'k-', linewidth=2)
        
        # Draw eyes
        right_eye = np.vstack([self.face_points[36:42], self.face_points[36]])
        left_eye = np.vstack([self.face_points[42:48], self.face_points[42]])
        self.ax.plot(right_eye[:, 0], right_eye[:, 1], 'k-', linewidth=2)
        self.ax.plot(left_eye[:, 0], left_eye[:, 1], 'k-', linewidth=2)
        
        # Draw mouth outer
        mouth_outer = np.vstack([self.face_points[48:60], self.face_points[48]])
        self.ax.plot(mouth_outer[:, 0], mouth_outer[:, 1], 'k-', linewidth=2)
        
        # Draw mouth inner
        mouth_inner = np.vstack([self.face_points[60:68], self.face_points[60]])
        self.ax.plot(mouth_inner[:, 0], mouth_inner[:, 1], 'k-', linewidth=2)
        
        # Update the canvas
        self.canvas.draw()
    
    def reset_face(self):
        # Reset all AU values to 0
        for au in self.action_units:
            self.action_units[au] = 0
            if au in self.sliders:
                self.sliders[au].set(0)
            if au in self.value_labels:
                self.value_labels[au].config(text="0")
        self.update_face()
    
    def set_expression(self, expression_name):
        # Reset all AUs first
        for au in self.action_units:
            self.action_units[au] = 0
            if au in self.sliders:
                self.sliders[au].set(0)
            if au in self.value_labels:
                self.value_labels[au].config(text="0")
        
        # Set values from predefined expression
        if expression_name in self.expressions:
            expression = self.expressions[expression_name]
            for au, value in expression.items():
                self.action_units[au] = value
                if au in self.sliders:
                    self.sliders[au].set(value)
                if au in self.value_labels:
                    self.value_labels[au].config(text=str(value))
        
        self.update_face()

def main():
    root = tk.Tk()
    app = FACSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()