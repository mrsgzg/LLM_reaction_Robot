import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, PathPatch, Ellipse
from matplotlib.path import Path
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from scipy.interpolate import splprep, splev

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
        # Simplified set of AUs for animation-style expressions
        self.action_units = {
            'AU1': 0,   # Inner Brow Raiser
            'AU2': 0,   # Outer Brow Raiser
            'AU4': 0,   # Brow Lowerer
            'AU5': 0,   # Upper Lid Raiser
            'AU6': 0,   # Cheek Raiser
            'AU12': 0,  # Lip Corner Puller (smile)
            'AU15': 0,  # Lip Corner Depressor
            'AU25': 0,  # Lips Part
            'AU26': 0,  # Jaw Drop
            'AU43': 0   # Eyes Closed
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
                'AU5': 40
            },
            'surprised': {
                'AU1': 70,
                'AU2': 70,
                'AU5': 80,
                'AU26': 70
            },
            'fearful': {
                'AU1': 60,
                'AU2': 60,
                'AU5': 70,
                'AU25': 40,
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
        ttk.Button(button_frame, text="Fearful", command=lambda: self.set_expression('fearful')).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=root.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Create right frame for AU controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Add title for AU controls
        ttk.Label(right_frame, text="Facial Action Units", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Create control frame for sliders
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize value_labels dictionary before creating sliders
        self.sliders = {}
        self.value_labels = {}
        
        # Create AU sliders
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
        
        # Add AU descriptions
        desc_frame = ttk.LabelFrame(right_frame, text="Action Unit Descriptions")
        desc_frame.pack(fill=tk.X, pady=10, padx=5)
        
        descriptions = [
            ("AU1", "Inner Brow Raiser"),
            ("AU2", "Outer Brow Raiser"),
            ("AU4", "Brow Lowerer"),
            ("AU5", "Upper Lid Raiser"),
            ("AU6", "Cheek Raiser"),
            ("AU12", "Lip Corner Puller (smile)"),
            ("AU15", "Lip Corner Depressor"),
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
    
    def smooth_curve(self, points, closed=False):
        """Create a smooth curve through points using spline interpolation"""
        if len(points) < 2:
            return None, None
            
        # Add tension to the curve by duplicating end points
        if not closed and len(points) > 2:
            # Extend the first and last points
            first_point = points[0]
            last_point = points[-1]
            
            # Calculate directions
            first_dir = points[0] - points[1]
            last_dir = points[-1] - points[-2]
            
            # Extend points
            extended_first = first_point + first_dir * 0.25
            extended_last = last_point + last_dir * 0.25
            
            extended_points = np.vstack([extended_first, points, extended_last])
        else:
            extended_points = points
        
        try:
            if closed:
                # For closed curves (like eyes), wrap around
                points_loop = np.vstack([extended_points, extended_points[0]])
                tck, u = splprep([points_loop[:, 0], points_loop[:, 1]], s=0, k=3, per=1)
            else:
                tck, u = splprep([extended_points[:, 0], extended_points[:, 1]], s=0, k=3)
            
            # Generate more points for a smoother curve
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)
            
            return x_new, y_new
        except Exception as e:
            print(f"Error creating smooth curve: {e}")
            # Fallback: return the original points
            return extended_points[:, 0], extended_points[:, 1]
    
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
        
        # Draw face with curves instead of lines and points
        self.draw_animated_face()
        
        # Update the canvas
        self.canvas.draw()
    
    def draw_animated_face(self):
        """Draw the face with a more cartoon/animated style"""
        # Background head shape (light skin tone)
        head_circle = Ellipse((170, 140), 150, 180, fill=True, 
                             facecolor='#FFE4C4', edgecolor=None, alpha=0.7)
        self.ax.add_patch(head_circle)
        
        # Draw eyebrows as smooth curves
        left_eyebrow = self.face_points[22:27]
        right_eyebrow = self.face_points[17:22]
        
        # Get smooth curves for eyebrows
        x_left_brow, y_left_brow = self.smooth_curve(left_eyebrow)
        x_right_brow, y_right_brow = self.smooth_curve(right_eyebrow)
        
        # Draw eyebrows with thicker lines
        if x_left_brow is not None and y_left_brow is not None:
            self.ax.plot(x_left_brow, y_left_brow, color='#5D4037', linewidth=4)
        if x_right_brow is not None and y_right_brow is not None:
            self.ax.plot(x_right_brow, y_right_brow, color='#5D4037', linewidth=4)
        
        # Draw eyes
        # Check how closed the eyes are based on AU43
        eye_closure = self.action_units['AU43'] / 100
        
        # Left eye (character's right)
        left_eye_center = np.mean(self.face_points[42:48], axis=0)
        left_eye_width = np.linalg.norm(self.face_points[45] - self.face_points[42]) * 1.2
        left_eye_height = np.linalg.norm(self.face_points[46] - self.face_points[44]) * (1.2 - eye_closure)
        
        # Get bezier curve for the eye shapes
        left_eye_points = self.face_points[42:48]
        left_eye_x, left_eye_y = self.smooth_curve(left_eye_points, closed=True)
        
        # Draw white of eye and pupil
        if eye_closure < 0.8:  # Only draw eye details if eyes are open enough
            if left_eye_x is not None and left_eye_y is not None:
                # White of eye
                self.ax.fill(left_eye_x, left_eye_y, color='white', edgecolor='black', linewidth=1.5)
                
                # Pupil
                pupil_size = max(left_eye_height * 0.6, 1)  # Ensure minimum size
                self.ax.add_patch(Circle(left_eye_center, pupil_size, color='black'))
                
                # Highlight/reflection
                highlight_size = pupil_size * 0.3
                highlight_pos = left_eye_center + np.array([pupil_size * 0.3, -pupil_size * 0.3])
                self.ax.add_patch(Circle(highlight_pos, highlight_size, color='white'))
        else:
            # Draw closed eye as a line
            self.ax.plot([left_eye_points[0, 0], left_eye_points[3, 0]], 
                         [left_eye_points[0, 1], left_eye_points[3, 1]], 
                         color='black', linewidth=2)
        
        # Right eye (character's left)
        right_eye_center = np.mean(self.face_points[36:42], axis=0)
        right_eye_width = np.linalg.norm(self.face_points[39] - self.face_points[36]) * 1.2
        right_eye_height = np.linalg.norm(self.face_points[40] - self.face_points[38]) * (1.2 - eye_closure)
        
        # Get bezier curve for the eye shapes
        right_eye_points = self.face_points[36:42]
        right_eye_x, right_eye_y = self.smooth_curve(right_eye_points, closed=True)
        
        # Draw white of eye and pupil
        if eye_closure < 0.8:  # Only draw eye details if eyes are open enough
            if right_eye_x is not None and right_eye_y is not None:
                # White of eye
                self.ax.fill(right_eye_x, right_eye_y, color='white', edgecolor='black', linewidth=1.5)
                
                # Pupil
                pupil_size = max(right_eye_height * 0.6, 1)  # Ensure minimum size
                self.ax.add_patch(Circle(right_eye_center, pupil_size, color='black'))
                
                # Highlight/reflection
                highlight_size = pupil_size * 0.3
                highlight_pos = right_eye_center + np.array([pupil_size * 0.3, -pupil_size * 0.3])
                self.ax.add_patch(Circle(highlight_pos, highlight_size, color='white'))
        else:
            # Draw closed eye as a line
            self.ax.plot([right_eye_points[0, 0], right_eye_points[3, 0]], 
                         [right_eye_points[0, 1], right_eye_points[3, 1]], 
                         color='black', linewidth=2)
        
        # Draw simplified nose
        nose_line = self.face_points[[27, 30]]
        self.ax.plot(nose_line[:, 0], nose_line[:, 1], color='black', linewidth=1.5)
        
        # Get smooth curves for mouth
        # Determine if mouth is open based on AU25 and AU26
        mouth_open = max(self.action_units['AU25'], self.action_units['AU26']) / 100
        
        # Outer mouth curve
        outer_mouth_points = np.vstack([self.face_points[48:55], self.face_points[55:60][::-1]])
        outer_x, outer_y = self.smooth_curve(outer_mouth_points, closed=True)
        
        # Inner mouth curve (only show if mouth is open)
        if mouth_open > 0.1:
            inner_mouth_points = self.face_points[60:68]
            inner_x, inner_y = self.smooth_curve(inner_mouth_points, closed=True)
        
        # Set mouth color based on expressions
        if self.action_units['AU12'] > 50:  # Smiling
            lip_color = '#FF5252'  # Brighter red for happy
        else:
            lip_color = '#D32F2F'  # Standard lip color
        
        # Draw mouth with smooth curves
        if outer_x is not None and outer_y is not None:
            if mouth_open > 0.1 and inner_x is not None and inner_y is not None:
                # If mouth is open, draw with inner mouth visible
                # Mouth interior (dark)
                self.ax.fill(inner_x, inner_y, color='#3E2723', edgecolor=None, alpha=0.7)
                
                # Draw lips as the outline
                self.ax.plot(outer_x, outer_y, color='black', linewidth=1.5)
            else:
                # Draw closed mouth as filled lips
                self.ax.fill(outer_x, outer_y, color=lip_color, edgecolor='black', linewidth=1.5)
        
        # Add cheeks if smiling (AU6 and AU12)
        if self.action_units['AU6'] > 30 or self.action_units['AU12'] > 50:
            intensity = max(self.action_units['AU6'], self.action_units['AU12']) / 100
            # Left cheek
            left_cheek_pos = self.face_points[41] + np.array([5, 15])
            self.ax.add_patch(Ellipse(left_cheek_pos, 20, 10, 
                                     fill=True, facecolor='#FFCDD2', 
                                     edgecolor=None, alpha=0.5 * intensity))
            
            # Right cheek
            right_cheek_pos = self.face_points[46] + np.array([-5, 15])
            self.ax.add_patch(Ellipse(right_cheek_pos, 20, 10, 
                                     fill=True, facecolor='#FFCDD2', 
                                     edgecolor=None, alpha=0.5 * intensity))
    
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
                if au in self.action_units:  # Check if this AU is in our simplified set
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