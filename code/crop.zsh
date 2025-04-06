#!/bin/zsh

# Input and Output Directories
input_dir="pull"
output_dir="output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Crop Parameters (width:height:x:y)
crop_settings="300:470:480:100"

# Trim Parameters (in frames)
start_frame=15
end_frame=$((start_frame + 10))

# Function to round time values
round_time() {
    printf "%.6f" "$1"
}

# Process all videos in the input directory
for video_file in "$input_dir"/*; do
    filename=$(basename "$video_file")

    echo "Processing: $filename"

    # Get FPS using ffprobe
    fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$video_file" | bc -l)

    # Calculate and round trim times in seconds
    start_time=$(round_time "$(echo "$start_frame / $fps" | bc -l)")
    end_time=$(round_time "$(echo "$end_frame / $fps" | bc -l)")

    # Output path
    output_file="$output_dir/$filename"

    # Crop + Trim using FFmpeg
    ffmpeg -i "$video_file" -vf "crop=${crop_settings}" -ss "$start_time" -to "$end_time" -c:v libx264 "$output_file"

    echo "Done: $output_file"
done

echo "All videos cropped and trimmed"


# ffmpeg -i "shombincoverdrive.mp4" -vf "crop=300:470:480:100" -ss "" -to "$end_time" -c:v libx264 "video.mp4"