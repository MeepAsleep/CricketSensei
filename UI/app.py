from flask import Flask, render_template, request, jsonify
import subprocess
import os
import json

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files or 'shot_type' not in request.form:
            return jsonify({"error": "Missing video file or shot type."})

        video = request.files['video']
        shot_type = request.form['shot_type']  # Get shot type from form

        if video.filename == '':
            return jsonify({"error": "No video file selected."})

        # Save uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        print(f"Video saved to: {video_path}")

        try:
            # Run comparison.py with video and shot_type arguments
            result = subprocess.run(
                ['python3', '../code/comparison.py', video_path, shot_type],
                capture_output=True, text=True, check=True
            )
            output = result.stdout
            print(f"Analysis output: {output}")

            # Redirect to results page with output and shot_type
            return jsonify({"output": output, "shot_type": shot_type})  

        except subprocess.CalledProcessError as e:
            output = f"An error occurred: {e.stderr}"
            print(f"Error during analysis: {output}")
            return jsonify({"error": output})  

    return render_template('index.html')

@app.route('/results')
def results():
    output = request.args.get('output', 'No output available.')
    shot_type = request.args.get('shot_type', 'default')

    # Splitting output based on phases
    feedback_dict = {"Starting Phase": [], "Middle Phase": [], "Ending Phase": []}
    current_phase = None

    for line in output.strip().split("\n"):
        line = line.strip()
        if "Starting Phase" in line:
            current_phase = "Starting Phase"
        elif "Middle Phase" in line:
            current_phase = "Middle Phase"
        elif "Ending Phase" in line:
            current_phase = "Ending Phase"
        elif current_phase and line.startswith("-"):
            feedback_dict[current_phase].append(line.strip("- "))

    return render_template('analysis.html', shot_type=shot_type, feedback=feedback_dict)



if __name__ == '__main__':
    app.run(debug=True)
