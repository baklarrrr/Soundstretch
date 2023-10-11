import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from moviepy.editor import AudioFileClip
from tkinter import messagebox
import soundfile as sf
import threading
import time

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soundstretch")

        self.audio_data = None
        self.sample_rate = None

        self.load_button = tk.Button(self.root, text="Load Audio", command=self.load_audio)
        self.load_button.pack()

        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack()

        self.speed_label = tk.Label(self.root, text="Speed:")
        self.speed_label.pack()
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_slider = tk.Scale(self.root, from_=0.03, to_=3.0, resolution=0.01, orient="horizontal", length=400, variable=self.speed_var)
        self.speed_slider.pack()

        self.selected_region = [None, None]  # [start, end]
        self.play_button = tk.Button(self.root, text="Play Selected", command=self.play_selected_audio)
        self.play_button.pack()

        self.gain_var = tk.DoubleVar(value=1.0)

        self.gain_label = tk.Label(self.root, text="Gain:")
        self.gain_label.pack()
        self.gain_slider = tk.Scale(self.root, from_=0.0, to_=10.0, resolution=0.01, orient="vertical", variable=self.gain_var, label="Gain")
        self.gain_slider.pack(side=tk.RIGHT)

        self.speed_entry = tk.Entry(self.root, textvariable=self.speed_var, width=5)
        self.speed_entry.pack(pady=5)
        self.speed_entry.bind('<Return>', lambda e: self.update_slider_from_entry())

        self.resolution_label = tk.Label(self.root, text="Resolution:")
        self.resolution_label.pack()
        self.resolution_var = tk.DoubleVar(value=1.0)
        self.resolution_slider = tk.Scale(self.root, from_=0.1, to_=10.0, resolution=0.1, orient="horizontal", variable=self.resolution_var, command=self.update_resolution)
        self.resolution_slider.pack()

        self.canvas.mpl_connect('button_press_event', self.on_middle_button_press)  # No button=2
        self.canvas.mpl_connect('button_release_event', self.on_middle_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_middle_button_motion)



        self.canvas.mpl_connect('scroll_event', self.on_zoom)

        self.panning = False
        self.start_pan_x = None




        # Button to export processed audio
        self.export_button = tk.Button(self.root, text="Export Processed Audio", command=self.export_audio)
        self.export_button.pack()

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Preset speed buttons
        self.speed_presets_frame = tk.Frame(self.root)
        self.speed_presets_frame.pack(pady=10)
        
        self.preset_speeds = [0.33, 0.5, 0.66, 1, 1.33, 1.66, 2]
        for speed in self.preset_speeds:
            button = tk.Button(self.speed_presets_frame, text=str(speed), command=lambda s=speed: self.set_speed(s))
            button.pack(side="left", padx=5)

    def update_resolution(self, event=None):
        if self.audio_data is not None:
            self.display_waveform()

    def update_slider_from_entry(self):
        try:
            new_speed = float(self.speed_entry.get())
            if 0.01 <= new_speed <= 10.0:
                self.speed_var.set(new_speed)
        except ValueError:
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, str(self.speed_var.get()))


    def set_speed(self, speed):
        """Set the slider value based on the preset speed button clicked."""
        self.speed_var.set(speed)

    def play_selected_audio(self):
        if not all(self.selected_region) or self.audio_data is None:
            return

        start_sample = librosa.time_to_samples(self.selected_region[0], sr=self.sample_rate)
        end_sample = librosa.time_to_samples(self.selected_region[1], sr=self.sample_rate)

        if start_sample == end_sample:
            return

        selected_audio = self.audio_data[start_sample:end_sample]

        # Check if selected_audio is empty
        if len(selected_audio) == 0:
            return

        time_stretched_audio = librosa.effects.time_stretch(selected_audio, rate=self.speed_var.get())

        # Remove the previous playback line if it exists
        if hasattr(self, 'playback_line'):
            self.playback_line.remove()

        # Initialize the playback line at the starting position
        self.playback_line, = self.ax.plot([self.selected_region[0], self.selected_region[0]], [np.min(selected_audio), np.max(selected_audio)], color='limegreen')

        # Start a new thread to update the playback line position
        threading.Thread(target=self.update_playback_line, args=(time_stretched_audio,)).start()

        # Play the time-stretched audio
        sd.play(time_stretched_audio * self.gain_var.get(), samplerate=self.sample_rate)



    def on_zoom(self, event):
        # Determine the zoom direction (in or out)
        base_scale = 1.5
        if event.button == 'up':
            # zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # zoom out
            scale_factor = base_scale
        else:
            # Not a recognized scroll event
            return

        # Get the current x-axis limits
        x1, x2 = self.ax.get_xlim()
    
        # Get the cursor position
        cursor_x = event.xdata

        if cursor_x is None:
            return
    
        # Set new x-axis limits based on the zoom direction and cursor position
        span = x2 - x1
        new_span = span * scale_factor
        x1_new = cursor_x - (cursor_x - x1) * scale_factor
        x2_new = cursor_x + (x2 - cursor_x) * scale_factor

        self.ax.set_xlim([x1_new, x2_new])
        self.canvas.draw()

    def on_middle_button_press(self, event):
        if event.button == 2:  # Ensure this is the middle button
            self.panning = True
            self.start_pan_x = event.xdata

    def on_middle_button_release(self, event):
        if event.button == 2:
            self.panning = False
            self.start_pan_x = None

    def on_middle_button_motion(self, event):
        if self.panning and self.start_pan_x is not None and event.xdata is not None:
            dx = self.start_pan_x - event.xdata
            xlim_left, xlim_right = self.ax.get_xlim()
            self.ax.set_xlim(xlim_left + dx, xlim_right + dx)
            self.canvas.draw_idle()




    def update_playback_line(self, audio_data):
        """Update the playback line position during playback."""
        playback_duration = len(audio_data) / self.sample_rate
        step_size = (self.selected_region[1] - self.selected_region[0]) / playback_duration
        current_x = self.selected_region[0]
    
        start_time = time.time()
    
        while current_x <= self.selected_region[1]:
            elapsed_time = time.time() - start_time
            if elapsed_time >= playback_duration:  # Stop updating if the playback is finished
                break
            current_x = self.selected_region[0] + (elapsed_time * step_size)
            self.playback_line.set_xdata([current_x, current_x])
            self.canvas.draw_idle()
            time.sleep(0.05)  # Sleep for a short duration before the next update


    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.mp4")])
        if not file_path:
            return

        if file_path.endswith(".mp4"):
            # Extract audio from the MP4 video
            with AudioFileClip(file_path) as clip:
                # Save audio to temporary WAV file, you can optimize this step to avoid writing to disk
                tmp_audio = "temp_audio.wav"
                clip.write_audiofile(tmp_audio, codec='pcm_s16le')
                try:
                    self.audio_data, self.sample_rate = librosa.load(tmp_audio, sr=None)
                except Exception as e:
                    messagebox.showerror("Error", str(e))
                os.remove(tmp_audio)  # Clean up temporary file
        else:
            try:
                self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)

                # Up-sample the audio data
                target_sample_rate = 100000  # Set your desired sample rate here
                self.audio_data = librosa.resample(self.audio_data, orig_sr=self.sample_rate, target_sr=target_sample_rate)
                self.sample_rate = target_sample_rate  # Update the sample rate

            except Exception as e:
                messagebox.showerror("Error", str(e))

        self.display_waveform()


    def display_waveform(self):
        self.ax.clear()
    
        # Resample data for visualization based on the resolution slider
        display_data = librosa.resample(self.audio_data, orig_sr=self.sample_rate, target_sr=int(self.sample_rate * self.resolution_var.get()))
    
        librosa.display.waveshow(display_data, sr=self.sample_rate * self.resolution_var.get(), ax=self.ax)
        self.canvas.draw()


    def on_click(self, event):
        if event.button == 1:  # Ensure this is the left button
            self.selected_region[0] = event.xdata

    def on_release(self, event):
        if event.button == 1:  # Ensure this is the left button
            self.selected_region[1] = event.xdata
            self.highlight_selected_region()


    def highlight_selected_region(self):
        if hasattr(self, 'highlighted_region'):
            self.highlighted_region.remove()  # Remove the previous highlighted region
        self.highlighted_region = self.ax.axvspan(self.selected_region[0], self.selected_region[1], color='red', alpha=0.5)
        self.canvas.draw()


    def export_audio(self):
        if self.audio_data is None or None in self.selected_region:
            tk.messagebox.showerror("Error", "Please load an audio file and select a region first.")
            return

        # Get the selected region from the audio data
        start_sample = librosa.time_to_samples(self.selected_region[0], sr=self.sample_rate)
        end_sample = librosa.time_to_samples(self.selected_region[1], sr=self.sample_rate)

        # Ensure start_sample and end_sample are different
        if start_sample == end_sample:
            tk.messagebox.showerror("Error", "Selected region is too small. Please select a valid region.")
            return

        # Slow down the selected region
        slowed_data = librosa.effects.time_stretch(y=self.audio_data[start_sample:end_sample], rate=self.speed_var.get())

        # Get the output directory from the user
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return  # User cancelled the directory selection

        # Save the processed audio as a .wav file using soundfile
        output_file_path = os.path.join(output_dir, "processed_audio.wav")
        sf.write(output_file_path, slowed_data, self.sample_rate)

        tk.messagebox.showinfo("Success", f"Processed audio saved to {output_file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()