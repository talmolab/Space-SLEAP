import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd

#Video Inventory Metadata.csv:
#Original File Name | Day | Recording Date | Location | Light Cycle Phase
VideoMeta = pd.read_csv('/Path/to/Video Inventory Metadata.csv') 
display(VideoMeta)

#Path to your parent DeepEthogram Folder
project_dir = "/Path/to/Deepethogram_Project_Folder"



#Align inference files to metadata and aggregate into one dataframe
prediction_files = glob.glob(os.path.join(project_dir, '**', '*_predictions.csv'), recursive=True)

associated_data = []

for file_path in prediction_files:
    # Extract the base name and remove the '_predictions.csv' suffix
    base_file_name_with_suffix = os.path.basename(file_path).replace('_predictions.csv', '')

    # Extract the part before the last underscore followed by 'Feeder' or 'Filter'
    base_file_name = base_file_name_with_suffix
    for suffix in ['_Feeder', '_Filter', '-Feeder', '_feeder']:
        base_file_name_parts = base_file_name.rsplit(suffix, 1)
        if len(base_file_name_parts) > 1:
            base_file_name = base_file_name_parts[0]
            break # Stop after finding the first matching suffix


    metadata_match = VideoMeta[VideoMeta['Original File Name'] == base_file_name]

    if not metadata_match.empty:
        # Assuming there's only one match per file name
        metadata_row = metadata_match.iloc[0]
        associated_data.append({
            'prediction_file': file_path,
            'Day': metadata_row['Day'],
            'Recording Date': metadata_row['Recording Date'],
            'Location': metadata_row['Location'],
            'Light Cycle Phase': metadata_row['Light Cycle Phase'],
            'original_file': base_file_name_with_suffix # Include the extracted original file name here
        })
    else:
        print(f"Metadata not found for file: {base_file_name_with_suffix}")


associated_df = pd.DataFrame(associated_data)

# Now that associated_df is created, proceed with processing prediction files
processed_data = []

for index, row in associated_df.iterrows():
    prediction_file_path = row['prediction_file']
    metadata = {
        'Day': row['Day'],
        'Light Cycle Phase': row['Light Cycle Phase'],
        'original_file': row['original_file']
    }

    try:
        prediction_df = pd.read_csv(prediction_file_path)

        # Assuming columns other than the first one are behavior probabilities
        behavior_columns = prediction_df.columns[1:].tolist()

        # Filter out non-numeric columns and the original frame index column if present
        numeric_behavior_columns = [col for col in behavior_columns if pd.api.types.is_numeric_dtype(prediction_df[col])]

        # Ensure the first column is treated as original_frame and is numeric
        prediction_df.iloc[:, 0] = pd.to_numeric(prediction_df.iloc[:, 0], errors='coerce')
        prediction_df.dropna(subset=[prediction_df.columns[0]], inplace=True) # Drop rows where original_frame couldn't be converted


        # Process only if there are valid numeric original_frame values
        if not prediction_df.empty:
            for _, pred_row in prediction_df.iterrows():
                original_frame = int(pred_row.iloc[0]) # Ensure original_frame is integer

                for behavior in numeric_behavior_columns:
                    # Consider behavior as occurring if probability > 0
                    if pred_row[behavior] > 0:
                        processed_data.append({
                            'original_frame': original_frame,
                            'predicted_behavior': behavior,
                            'Day': metadata['Day'],
                            'Light Cycle Phase': metadata['Light Cycle Phase'],
                            'original_file': metadata['original_file']
                        })
        else:
            print(f"Skipping file {prediction_file_path} due to no valid original_frame data.")


    except Exception as e:
        print(f"Error processing file {prediction_file_path}: {e}")
        # Continue to the next file even if one fails

# Convert the list of processed data into a DataFrame
processed_df = pd.DataFrame(processed_data)

# Display the head and info of the processed DataFrame to verify
display(processed_df.head())
display(processed_df.info())



#Generate a global frame index
# Define categorical order for Light Cycle Phase to ensure correct sorting
light_cycle_order = pd.CategoricalDtype(['Light', 'Dark'], ordered=True)
processed_df['Light Cycle Phase'] = processed_df['Light Cycle Phase'].astype(light_cycle_order)

# Sort the DataFrame by Day and Light Cycle Phase
processed_df_sorted = processed_df.sort_values(by=['Day', 'Light Cycle Phase', 'original_file', 'original_frame']).reset_index(drop=True)

# Initialize global frame index
processed_df_sorted['global_frame_index'] = 0

current_global_frame = 0
# Iterate through each unique video segment and calculate global frame index
for name, group in processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'], sort=False):
    # Get the original frame indices for the current segment
    original_frames = group['original_frame'].values

    # Add a check to ensure original_frames is not empty before proceeding
    if original_frames.size > 0:
        # Calculate the global frame index for the current segment
        global_frames = original_frames + current_global_frame

        # Assign the calculated global frame indices back to the DataFrame
        # Ensure we are assigning to the correct rows corresponding to the group
        processed_df_sorted.loc[group.index, 'global_frame_index'] = global_frames

        # Update the current_global_frame for the next segment
        # The number of frames in the current segment is the max original frame + 1 (assuming 0-based indexing)
        frames_in_segment = original_frames.max() + 1
        current_global_frame += frames_in_segment
    else:
        print(f"Warning: Empty group found for {name}. Global frame index not updated.") # Debugging message


# Display the head of the updated DataFrame, check its data type, and check for nulls in global_frame_index
display(processed_df_sorted.head())
display(processed_df_sorted['global_frame_index'].dtype)
display(processed_df_sorted['global_frame_index'].isnull().sum())



#Define a color palette for plots
all_behaviors = sorted(processed_df_sorted['predicted_behavior'].unique())

# Define the color palette
custom_colors = ['#580E3C','#264653','#2A9D8F','#8AB17D','#E9C46A','#F4A261','#E76F51','#B43718'] #include color for background
palette = sns.color_palette(custom_colors, len(all_behaviors))

# Create a dictionary mapping each behavior to a color
behavior_colors = {behavior: color for behavior, color in zip(all_behaviors, palette)}

# Display the behavior-color mapping
print("Behavior Color Mapping:")
display(behavior_colors)



#Plot an inference only database-wide ethogram aligned to light cycle
all_behaviors = sorted(processed_df['predicted_behavior'].unique())
behavior_mapping = {behavior: i for i, behavior in enumerate(all_behaviors)}

print("Behavior mapping for y-axis:")
display(behavior_mapping)

# Identify the global frame indices that correspond to the start of each new day
# This is the minimum global_frame_index for each unique 'Day'
day_boundaries = processed_df_sorted.groupby('Day')['global_frame_index'].min().reset_index()

# Create a figure with two subplots: one for the raster plot and one for the light cycle
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 5), gridspec_kw={'height_ratios': [25, 1]}, sharex=True) # Adjust figure size and height ratios as needed

# Use a colormap to get unique colors for each behavior
#colors = plt.cm.get_cmap('tab10', len(behavior_mapping)) # Remove or comment out this line

# Plot the raster plot on the first subplot (ax1)
for behavior, y_pos in behavior_mapping.items():
    # Filter data for the current behavior
    behavior_data = processed_df_sorted[processed_df_sorted['predicted_behavior'] == behavior]

    if not behavior_data.empty:
        # Plot vertical lines for each global frame index where the behavior occurs
        ax1.vlines(x=behavior_data['global_frame_index'],
                  ymin=y_pos - 0.4, # Start of the behavior band
                  ymax=y_pos + 0.4, # End of the behavior band
                  color=behavior_colors[behavior],
                  linewidth=1) # Adjust linewidth as needed

# Set y-axis ticks and labels for the raster plot
ax1.set_yticks(list(behavior_mapping.values()))
ax1.set_yticklabels(list(behavior_mapping.keys()))
ax1.set_ylabel("Behavior", fontsize=14)
ax1.set_title("Behavioral Ethogram Across All Experiment Days", fontsize=20)
ax1.set_ylim(-0.5, len(behavior_mapping) - 0.5)

# Add vertical lines at day boundaries on the raster plot
for index, row in day_boundaries.iterrows():
    ax1.axvline(x=row['global_frame_index'], color='gray', linestyle='--', linewidth=1.5)
    # Add day label near the boundary line, adjust y position further down
    ax1.text(row['global_frame_index'], -1, f"Day {int(row['Day'])}", rotation=45, ha='right', va='top', fontsize=10, color='black') # Adjusted y position


# Plot the light cycle on the second subplot (ax2)
# We need to determine the start and end global frame index for each light cycle phase segment
light_cycle_min = processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'])['global_frame_index'].min().reset_index(name='min_global_frame')
light_cycle_max = processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'])['global_frame_index'].max().reset_index(name='max_global_frame')

light_cycle_segments = pd.merge(light_cycle_min, light_cycle_max,
                                on=['Day', 'Light Cycle Phase', 'original_file'],
                                how='left').reset_index() # Reset index to make grouped columns regular columns

# Define colors for light and dark cycles
light_cycle_colors = {'Light': 'yellow', 'Dark': 'darkblue'}

# Plot rectangles for each light cycle segment
for index, row in light_cycle_segments.iterrows():
    # Check for NaN values in min/max global frame before plotting
    if pd.notnull(row['min_global_frame']) and pd.notnull(row['max_global_frame']):
        start_frame = row['min_global_frame']
        end_frame = row['max_global_frame']
        # Access 'Light Cycle Phase' safely using .get() or by checking column existence
        light_cycle = row.get('Light Cycle Phase', None) # Use .get() with a default
        if light_cycle is None and 'Light Cycle Phase' in row.index: # Fallback check if .get() somehow fails
             light_cycle = row['Light Cycle Phase']

        color = light_cycle_colors.get(light_cycle, 'gray') # Default to gray if phase is not Light or Dark or None

        ax2.add_patch(plt.Rectangle((start_frame, 0), end_frame - start_frame, 0.2, color=color)) # Decreased height to 0.5


# Set y-axis for the light cycle plot
ax2.set_yticks([0.1]) # Adjust tick position to be in the middle of the shorter bar
ax2.set_yticklabels(['Light Cycle'])
ax2.set_ylim(0, 0.2) # Adjust y-axis limit to fit the shorter bar
#ax2.set_xlabel("Global Frame Index", fontsize=16) # Set x-axis label on the lower subplot

# Set the x-axis limits to tightly fit the data for both subplots (shared x-axis)
ax1.set_xlim(processed_df_sorted['global_frame_index'].min(), processed_df_sorted['global_frame_index'].max())

ax1.set_xticklabels([])
ax1.set_xticks([])
ax2.set_xticklabels([])
ax2.set_xticks([])

# Adjust space between subplots
plt.subplots_adjust(hspace=0) # Set hspace to 0 to remove vertical space

# Create legends for both plots
# Legend for behaviors
legend_handles_behaviors = [mpatches.Patch(color=behavior_colors[behavior], label=behavior) for behavior in behavior_mapping.keys()] # Use behavior_colors for legend patches
ax1.legend(handles=legend_handles_behaviors, title="Behaviors", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

# Legend for light cycle
legend_handles_lightcycle = [mpatches.Patch(color=color, label=phase) for phase, color in light_cycle_colors.items()]
ax2.legend(handles=legend_handles_lightcycle, title="Light Cycle", bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=12)


plt.show()



#Sum data for cumulative Measures
behavior_counts = processed_df_sorted['predicted_behavior'].value_counts().reset_index()
behavior_counts.columns = ['predicted_behavior', 'total_frames']
display(behavior_counts)



#Bar graph of total frames per behavior
plt.figure(figsize=(12, 6))
sns.barplot(x='predicted_behavior', y='total_frames', data=behavior_counts,
            hue='predicted_behavior', palette=behavior_colors, legend=False)
sns.despine(top=True, right=True)
plt.title('Total Frames per Behavior', fontsize=17)
plt.xlabel('Behavior', fontsize=16)
plt.ylabel('Total Frames', fontsize=16)
plt.xticks(rotation=0, ha='center', fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()



#Part of a whole horizontal bar graph of total frames per behavior
plt.figure(figsize=(20, 3))

# Sort behaviors by total frames for better visualization in the stack
behavior_counts_sorted = behavior_counts.sort_values(by='total_frames', ascending=False).reset_index(drop=True)

cumulative_frames = 0
for index, row in behavior_counts_sorted.iterrows():
    behavior = row['predicted_behavior']
    frames = row['total_frames']
    color = behavior_colors[behavior]

    # Plot each behavior as a segment of the stacked bar
    plt.barh(y=0, width=frames, left=cumulative_frames, color=color, label=behavior) # Use barh for horizontal bars

    # Update cumulative frames for the next bar segment
    cumulative_frames += frames


#plt.title('Total Frames per Behavior')
plt.xlabel('Total Frames', fontsize=26)
plt.xticks(fontsize=24)
plt.ylabel('') # No y-label needed for a single stacked bar
plt.yticks([]) # Remove y-axis ticks for a single stacked bar
#plt.legend(title='Behaviors', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap
plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to prevent legend overlap

# Remove the top and right spines (axes)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False) # Remove the left spine (y-axis line)
plt.yticks([]) # Ensure no y-axis ticks remain

# Set the x-axis limits to tightly fit the data
plt.xlim(0, cumulative_frames)


plt.show()



#Sum data for light cycle-based cumulative data
# Group by Light Cycle Phase and behavior_name and count the total frames
behavior_counts_lightcycle = processed_df_sorted.groupby(['Light Cycle Phase', 'predicted_behavior'], observed=False).size().reset_index(name='total_frames')

# Display the resulting DataFrame
display(behavior_counts_lightcycle)



#Plot bar graph of cumulative behaviors by light cycle
# Reuse the behavior_mapping and colors from the ethogram plot
all_behaviors = sorted(behavior_counts_lightcycle['predicted_behavior'].unique())
behavior_mapping = {behavior: i for i, behavior in enumerate(all_behaviors)}
colors = plt.cm.get_cmap('tab10', len(behavior_mapping))

plt.figure(figsize=(12, 6))
sns.barplot(x='predicted_behavior', y='total_frames', hue='Light Cycle Phase', data=behavior_counts_lightcycle, palette={'Light': 'yellow', 'Dark': 'darkblue'})

plt.title('Total Frames per Behavior Across Light Cycle Phases', fontsize=17)
plt.xlabel('Behavior', fontsize=16)
plt.ylabel('Total Frames', fontsize=16)
plt.xticks(rotation=0, ha='center', fontsize=14)

plt.tight_layout()
plt.show()



#Plot bar graph of cummulative behavior as a percent of all frames by light cycle
# Calculate total frames for each light cycle phase
total_frames_per_lightcycle = processed_df_sorted.groupby('Light Cycle Phase', observed=False).size().reset_index(name='total_lightcycle_frames')

# Group by Light Cycle Phase and predicted_behavior and count the total frames
behavior_counts_lightcycle = processed_df_sorted.groupby(['Light Cycle Phase', 'predicted_behavior'], observed=False).size().reset_index(name='total_frames')

# Merge the behavior counts with the total frames per light cycle phase
behavior_percentage_lightcycle = pd.merge(behavior_counts_lightcycle, total_frames_per_lightcycle, on='Light Cycle Phase', how='left')

# Calculate the percentage of frames for each behavior within its light cycle phase
behavior_percentage_lightcycle['percentage'] = (behavior_percentage_lightcycle['total_frames'] / behavior_percentage_lightcycle['total_lightcycle_frames']) * 100

# Reuse the behavior_mapping and colors from the ethogram plot
all_behaviors = sorted(behavior_percentage_lightcycle['predicted_behavior'].unique())
behavior_mapping = {behavior: i for i, behavior in enumerate(all_behaviors)}
colors = plt.cm.get_cmap('tab10', len(behavior_mapping))

plt.figure(figsize=(12, 6))

sns.barplot(x='predicted_behavior', y='percentage', hue='Light Cycle Phase', data=behavior_percentage_lightcycle, palette={'Light': 'yellow', 'Dark': 'darkblue'})
sns.despine(top=True, right=True)

plt.title('Percentage of Frames per Behavior Across Light Cycle Phases', fontsize=17)
plt.xlabel('Behavior', fontsize=16)
plt.ylabel('Percent Frames', fontsize=16)
plt.xticks(rotation=0, ha='center', fontsize=14)

plt.tight_layout()
plt.show()



#Plot parth of whole horizontal bar graph of cumulative behaviors 
# Create a figure with two subplots, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True) # Adjusted figure size

# Define the light cycle phases
light_cycle_phases = ['Light', 'Dark']

# Determine a consistent order for behaviors based on total frames across both phases
# This ensures the stacked bars have the same behavior segments in the same order
total_behavior_counts = processed_df_sorted.groupby('predicted_behavior').size().reset_index(name='total_frames')
consistent_behavior_order = total_behavior_counts.sort_values(by='total_frames', ascending=False)['predicted_behavior'].tolist()

for i, phase in enumerate(light_cycle_phases):
    # Filter data for the current light cycle phase
    phase_df = processed_df_sorted[processed_df_sorted['Light Cycle Phase'] == phase].copy()

    # Calculate total frames per behavior for the current phase
    behavior_counts_phase = phase_df['predicted_behavior'].value_counts().reset_index()
    behavior_counts_phase.columns = ['predicted_behavior', 'total_frames']

    # Reindex behavior_counts_phase to match the consistent_behavior_order and fill missing behaviors with 0
    behavior_counts_phase = behavior_counts_phase.set_index('predicted_behavior').reindex(consistent_behavior_order).fillna(0).reset_index()


    cumulative_frames = 0
    # Determine which axis to plot on
    ax = ax1 if phase == 'Light' else ax2

    # Plot in the consistent behavior order
    for index, row in behavior_counts_phase.iterrows():
        behavior = row['predicted_behavior']
        frames = row['total_frames']
        color = behavior_colors[behavior] # Use the color from the universal mapping

        # Plot each behavior as a segment of the stacked bar on the correct axis
        ax.barh(y=0, width=frames, left=cumulative_frames, color=color, label=behavior) # Use barh for horizontal bars

        # Update cumulative frames for the next bar segment
        cumulative_frames += frames

    # Set title for each subplot
    ax.set_title('')
    ax.set_ylabel(f'{phase} cycle', fontsize = 20)
    ax.set_yticks([]) # Remove y-axis ticks for a single stacked bar


    # Remove the top and right spines (axes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Remove the left spine (y-axis line)
    ax.set_yticks([]) # Ensure no y-axis ticks remain

ax1.set_xlabel('')
ax1.set_title('Total Frames per Behavior Across Light Cycle Phases', fontsize=28)

ax2.set_xlabel('Total Frames',fontsize=26)
ax2.tick_params(axis='x', labelsize=24) # Adjust x-tick font size

# Set common x-axis label
#fig.text(0.5, 0.04, 'Total Frames', ha='center', fontsize=26) # Adjusted position

# Create a single legend for all behaviors using the consistent order
#legend_handles = [mpatches.Patch(color=colors(behavior_mapping[b]), label=b) for b in consistent_behavior_order]
#fig.legend(handles=legend_handles, title='Behaviors', bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout(rect=[0, 0.03, 0.9, 1]) # Adjust layout to prevent legend overlap and make space for x-label
#plt.title('Total Frames per Behavior Across Light Cycle Phases', fontsize=17)

# Set the x-axis limits to match the maximum cumulative frames across both phases
max_cumulative_frames_light = processed_df_sorted[processed_df_sorted['Light Cycle Phase'] == 'Light'].shape[0]
max_cumulative_frames_dark = processed_df_sorted[processed_df_sorted['Light Cycle Phase'] == 'Dark'].shape[0]
max_cumulative_frames = max(max_cumulative_frames_light, max_cumulative_frames_dark)

ax1.set_xlim(0, max_cumulative_frames)
ax2.set_xlim(0, max_cumulative_frames)


plt.show()



#Plot line graph of longitudinal cummulative behaviors
# Calculate the cumulative counts of each behavior over the global frame index
# First, create a DataFrame that includes all global frame indices for each behavior
all_frames = pd.DataFrame({'global_frame_index': processed_df_sorted['global_frame_index'].unique()})
all_frames['key'] = 0

cumulative_df = pd.DataFrame()

# Iterate through each behavior to calculate cumulative counts
for behavior in processed_df_sorted['predicted_behavior'].unique():
    # Filter data for the current behavior
    behavior_data = processed_df_sorted[processed_df_sorted['predicted_behavior'] == behavior].copy()

    # Add a count column for this behavior (1 for each occurrence)
    behavior_data['count'] = 1

    # Merge with all_frames to ensure all frames are considered, filling missing counts with 0
    merged_df = pd.merge(all_frames, behavior_data[['global_frame_index', 'count']], on='global_frame_index', how='left').fillna(0)

    # Calculate the cumulative sum of counts
    merged_df[f'cumulative_{behavior}'] = merged_df['count'].cumsum()

    # Select and rename the cumulative column, and add it to cumulative_df
    if cumulative_df.empty:
        cumulative_df = merged_df[['global_frame_index', f'cumulative_{behavior}']]
    else:
        cumulative_df = pd.merge(cumulative_df, merged_df[['global_frame_index', f'cumulative_{behavior}']], on='global_frame_index', how='left')

# Sort by global_frame_index
cumulative_df = cumulative_df.sort_values(by='global_frame_index').reset_index(drop=True)

# Plot the cumulative behaviors
plt.figure(figsize=(33, 5))

for behavior in all_behaviors:
    plt.plot(cumulative_df['global_frame_index'], cumulative_df[f'cumulative_{behavior}'],
             label=behavior, color=behavior_colors[behavior], linewidth=3) # Corrected color assignment

# Add vertical lines at day boundaries
day_boundaries = processed_df_sorted.groupby('Day')['global_frame_index'].min().reset_index()
for index, row in day_boundaries.iterrows():
    plt.axvline(x=row['global_frame_index'], color='gray', linestyle='--', linewidth=1.5)

# Set x-axis ticks and labels to show day boundaries below the axis
plt.xticks(day_boundaries['global_frame_index'], [f"{int(d)}" for d in day_boundaries['Day']], rotation=0, ha='center')

plt.title('Cumulative Behavior Counts Over Time', fontsize=17)
plt.xlabel('Day')
plt.ylabel('Cumulative Frame Count', fontsize=14)
plt.legend(title='Behavior', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to prevent legend overlap
#plt.tight_layout(rect=[0, 0, 1, 1])

# Eliminate blank spaces before and after x-axis values
plt.xlim(cumulative_df['global_frame_index'].min(), cumulative_df['global_frame_index'].max())



plt.show()



#Calculate durations of each behavior bout
behavior_segments = []

# Get unique behaviors
unique_behaviors = processed_df_sorted['predicted_behavior'].unique()

for behavior in unique_behaviors:
    # Filter the DataFrame for the current behavior
    behavior_df = processed_df_sorted[processed_df_sorted['predicted_behavior'] == behavior].copy()

    # Sort by original_file and global_frame_index to ensure correct order within each file
    behavior_df_sorted = behavior_df.sort_values(by=['original_file', 'global_frame_index']).reset_index(drop=True)

    current_segment = None

    for index, row in behavior_df_sorted.iterrows():
        current_global_frame_index = row['global_frame_index']
        current_file = row['original_file']

        if current_segment is None:
            # Start the first segment for this behavior and file
            current_segment = {
                'behavior': behavior,
                'start_frame': current_global_frame_index,
                'end_frame': current_global_frame_index,
                'original_file': current_file
            }
        else:
            # Check if the current row continues the current segment for this behavior and file
            is_consecutive_frame = current_global_frame_index == current_segment['end_frame'] + 1
            is_same_file = current_file == current_segment['original_file']

            if is_consecutive_frame and is_same_file:
                # Extend the current segment
                current_segment['end_frame'] = current_global_frame_index
            else:
                # Close the previous segment and start a new one for this behavior and file
                behavior_segments.append(current_segment)
                current_segment = {
                    'behavior': behavior,
                    'start_frame': current_global_frame_index,
                    'end_frame': current_global_frame_index,
                    'original_file': current_file
                }

    # Add the last segment for this behavior after the loop finishes
    if current_segment is not None:
        behavior_segments.append(current_segment)


# Convert the list of segments into a DataFrame for easier handling
behavior_segments_df = pd.DataFrame(behavior_segments)

# Calculate the duration of each segment
behavior_segments_df['duration'] = behavior_segments_df['end_frame'] - behavior_segments_df['start_frame'] + 1

# Display the head of the segments DataFrame to verify
display(behavior_segments_df.head())
display(behavior_segments_df.info())


behavior_durations = {}

for index, row in behavior_segments_df.iterrows():
    behavior = row['behavior']
    duration = row['duration']

    if behavior not in behavior_durations:
        behavior_durations[behavior] = []

    behavior_durations[behavior].append(duration)



#Plot distributions of behavioral bouts
sns.set_context("talk")

# Reuse the behavior_colors dictionary for consistent coloring
# No need to create a new palette here

Hists = sns.FacetGrid(behavior_segments_df, row="behavior", hue="behavior",
                    aspect=4, height=1.5, palette=behavior_colors, # Pass the entire dictionary as palette
                    subplot_kws={"facecolor": (0, 0, 0, 0)})

Hists.map(sns.kdeplot, "duration", bw_adjust=0.2,
          clip_on=False,
          fill=True, alpha=1, linewidth=1.5, clip=(0,200))
Hists.map(sns.kdeplot, "duration", bw_adjust=0.2, clip_on=False, color="w", lw=2.5, clip=(0, 200))

Hists.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

def label(x, color, label):
  ax = plt.gca()
  # Use the correct color for the label from the behavior_colors dictionary
  ax.text(1, .3, label, fontweight="bold", color=behavior_colors[label],
          ha="right", va="center", transform=ax.transAxes)

Hists.map(label, "behavior")

Hists.figure.subplots_adjust(hspace=-.25)

Hists.set_titles("")
Hists.set(yticks=[], ylabel="")
Hists.despine(bottom=True, left=True)
Hists.set(xlabel="Duration (frames)")
Hists.figure.suptitle("Duration Distribution for Each Behavior", ha="center", x=0.6);



#==========Make longitudinal ethogram of both labels and inferences, stacked=========



#Align both labels and predictions to metadata
prediction_files = glob.glob(os.path.join(project_dir, '**', '*_predictions.csv'), recursive=True)
label_files = glob.glob(os.path.join(project_dir, '**', '*_labels.csv'), recursive=True)

associated_data = []

# Process prediction files
for file_path in prediction_files:
    # Extract the base name and remove the '_predictions.csv' suffix
    base_file_name_with_suffix = os.path.basename(file_path).replace('_predictions.csv', '')

    # Extract the part before the last underscore followed by 'Feeder' or 'Filter'
    base_file_name = base_file_name_with_suffix
    for suffix in ['_Feeder', '_Filter', '-Feeder', '_feeder']:
        base_file_name_parts = base_file_name.rsplit(suffix, 1)
        if len(base_file_name_parts) > 1:
            base_file_name = base_file_name_parts[0]
            break # Stop after finding the first matching suffix


    metadata_match = VideoMeta[VideoMeta['Original File Name'] == base_file_name]

    if not metadata_match.empty:
        # Assuming there's only one match per file name
        metadata_row = metadata_match.iloc[0]
        associated_data.append({
            'file_path': file_path,
            'data_type': 'prediction', # Add a column to distinguish data type
            'Day': metadata_row['Day'],
            'Recording Date': metadata_row['Recording Date'],
            'Location': metadata_row['Location'],
            'Light Cycle Phase': metadata_row['Light Cycle Phase'],
            'original_file': base_file_name_with_suffix # Include the extracted original file name here
        })
    else:
        print(f"Metadata not found for prediction file: {base_file_name_with_suffix}")

# Process label files
for file_path in label_files:
    # Extract the base name and remove the '_labels.csv' suffix
    base_file_name_with_suffix = os.path.basename(file_path).replace('_labels.csv', '')

    # Extract the part before the last underscore followed by 'Feeder' or 'Filter'
    base_file_name = base_file_name_with_suffix
    for suffix in ['_Feeder', '_Filter', '-Feeder', '_feeder']:
        base_file_name_parts = base_file_name.rsplit(suffix, 1)
        if len(base_file_name_parts) > 1:
            base_file_name = base_file_name_parts[0]
            break # Stop after finding the first matching suffix


    metadata_match = VideoMeta[VideoMeta['Original File Name'] == base_file_name]

    if not metadata_match.empty:
        # Assuming there's only one match per file name
        metadata_row = metadata_match.iloc[0]
        associated_data.append({
            'file_path': file_path,
            'data_type': 'label', # Add a column to distinguish data type
            'Day': metadata_row['Day'],
            'Recording Date': metadata_row['Recording Date'],
            'Location': metadata_row['Location'],
            'Light Cycle Phase': metadata_row['Light Cycle Phase'],
            'original_file': base_file_name_with_suffix # Include the extracted original file name here
        })
    else:
        print(f"Metadata not found for label file: {base_file_name_with_suffix}")


associated_df = pd.DataFrame(associated_data)

# --- Debugging Step: Display associated_df ---
print("\nAssociated DataFrame:")
display(associated_df.head())
display(associated_df['data_type'].value_counts())
# --- End Debugging Step ---

# Now that associated_df is created, proceed with processing each file
processed_data = []

for index, row in associated_df.iterrows():
    file_path = row['file_path']
    data_type = row['data_type']
    metadata = {
        'Day': row['Day'],
        'Light Cycle Phase': row['Light Cycle Phase'],
        'original_file': row['original_file'],
        'data_type': data_type # Include data_type in metadata for processing
    }

    try:
        current_df = pd.read_csv(file_path)

        # Ensure the first column is treated as original_frame and is numeric
        current_df.iloc[:, 0] = pd.to_numeric(current_df.iloc[:, 0], errors='coerce')
        current_df.dropna(subset=[current_df.columns[0]], inplace=True) # Drop rows where original_frame couldn't be converted

        # Assuming columns other than the first one are behavior probabilities/labels
        behavior_columns = current_df.columns[1:].tolist()

        # Process only if there are valid numeric original_frame values
        if not current_df.empty:
            for _, data_row in current_df.iterrows():
                original_frame = data_row.iloc[0] # Keep original_frame as numeric before int conversion

                # Ensure original_frame is not NaN and is an integer before proceeding
                if pd.notna(original_frame) and original_frame == int(original_frame):
                    original_frame_int = int(original_frame)

                    for behavior_col in behavior_columns: # Iterate through all behavior columns
                        # For prediction data, consider behavior as occurring if probability > 0
                        # For label data, consider behavior as occurring if value is 1 (or > 0 if it's probability)
                        # Add a check to ensure the behavior column exists and is numeric before accessing
                        if behavior_col in data_row and pd.api.types.is_numeric_dtype(pd.Series(data_row[behavior_col])):
                             if data_type == 'prediction' and data_row[behavior_col] > 0:
                                  processed_data.append({
                                      'original_frame': original_frame_int,
                                      'behavior': behavior_col, # Store behavior name in a single column
                                      'data_type': data_type,
                                      'Day': metadata['Day'],
                                      'Light Cycle Phase': metadata['Light Cycle Phase'],
                                      'original_file': metadata['original_file']
                                  })
                             elif data_type == 'label' and data_row[behavior_col] > 0: # Assuming label values are 1 for presence
                                  processed_data.append({
                                      'original_frame': original_frame_int,
                                      'behavior': behavior_col, # Store behavior name in a single column
                                      'data_type': data_type,
                                      'Day': metadata['Day'],
                                      'Light Cycle Phase': metadata['Light Cycle Phase'],
                                      'original_file': metadata['original_file']
                                  })
                        # Optionally, add a message if a behavior column is missing or not numeric
                        # else:
                        #     print(f"Warning: Behavior column '{behavior_col}' missing or not numeric in file {file_path}")


        else:
            print(f"Skipping processing file {file_path} due to no valid original_frame data.")


    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        # Continue to the next file even if one fails

# Convert the list of processed data into a DataFrame
# This DataFrame will have rows for both predicted and labeled behaviors
processed_df = pd.DataFrame(processed_data)

# Rename the 'behavior' column to make it clear
processed_df.rename(columns={'behavior': 'behavior_name'}, inplace=True)


# Display the head and info of the processed DataFrame to verify
display(processed_df.head())
display(processed_df.info())
display(processed_df['data_type'].value_counts())



#Create a new global frame index that overlaps index for the same name files regardless of label or prediction
# Define categorical order for Light Cycle Phase to ensure correct sorting
light_cycle_order = pd.CategoricalDtype(['Light', 'Dark'], ordered=True)
# Ensure 'Light Cycle Phase' exists before attempting to categorize
if 'Light Cycle Phase' in processed_df.columns:
    processed_df['Light Cycle Phase'] = processed_df['Light Cycle Phase'].astype(light_cycle_order)


# Sort the DataFrame by Day, Light Cycle Phase, original_file, and original_frame
processed_df_sorted = processed_df.sort_values(by=['Day', 'Light Cycle Phase', 'original_file', 'original_frame']).reset_index(drop=True)

# Ensure 'original_frame' is numeric before calculations
processed_df_sorted['original_frame'] = pd.to_numeric(processed_df_sorted['original_frame'], errors='coerce')

# Initialize global frame index
processed_df_sorted['global_frame_index'] = 0

current_global_frame = 0
# Iterate through each unique video segment (defined by Day, Light Cycle Phase, original_file) and calculate global frame index
for name, group in processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'], sort=False):
    # Get the original frame indices for the current segment, dropping NaNs
    original_frames = group['original_frame'].dropna().values

    # Add a check to ensure original_frames is not empty before proceeding
    if original_frames.size > 0:
        try:
            # Calculate the global frame index for the current segment
            global_frames = original_frames + current_global_frame

            # Assign the calculated global frame indices back to the DataFrame
            # Ensure we are assigning to the correct rows corresponding to the group and valid original_frames
            valid_indices = group['original_frame'].dropna().index
            processed_df_sorted.loc[valid_indices, 'global_frame_index'] = global_frames

            # Update the current_global_frame for the next segment
            # The number of frames in the current segment is the max original frame + 1 (assuming 0-based indexing)
            frames_in_segment = original_frames.max() + 1
            current_global_frame += frames_in_segment
        except Exception as e:
            print(f"Error calculating global frame index for group {name}: {e}")
            # If an error occurs, print a warning but still try to update current_global_frame if possible
            if original_frames.size > 0:
                 try:
                      frames_in_segment = original_frames.max() + 1
                      current_global_frame += frames_in_segment
                 except:
                      print(f"Could not update current_global_frame for group {name}.")
            print(f"Warning: Skipping global frame index update for some rows in group {name} due to error.")
    else:
        print(f"Warning: Empty or all-NaN 'original_frame' values in group for {name}. Global frame index not updated for this group.")


# Display the head of the updated DataFrame, check its data type, and check for nulls in global_frame_index
display(processed_df_sorted.head())



#Plot both labels and inferences ehtogram stacked
# Identify the global frame indices that correspond to the start of each new day
day_boundaries = processed_df_sorted.groupby('Day')['global_frame_index'].min().reset_index()

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 12), gridspec_kw={'height_ratios': [20, 20, 0.8]}, sharex=True) # Adjust figure size and height ratios as needed

# Plot Labeled Behaviors on the first subplot (ax1)
for behavior, y_pos in behavior_mapping.items():
    # Filter data for the current labeled behavior
    behavior_data = processed_df_sorted[(processed_df_sorted['behavior_name'] == behavior) & (processed_df_sorted['data_type'] == 'label')]

    if not behavior_data.empty:
        # Plot vertical lines for each global frame index where the labeled behavior occurs
        ax1.vlines(x=behavior_data['global_frame_index'],
                  ymin=y_pos - 0.4,
                  ymax=y_pos + 0.4,
                  color=behavior_colors[behavior],
                  linewidth=1)

# Set y-axis ticks and labels for the labeled behavior plot
ax1.set_yticks(list(behavior_mapping.values()))
ax1.set_yticklabels(list(behavior_mapping.keys()))
ax1.set_ylabel("Labeled Behavior", fontsize=14)
ax1.set_title("Behavioral Ethogram Across All Experiment Days", fontsize=16)
ax1.set_ylim(-0.5, len(behavior_mapping) - 0.5)
ax1.grid(axis='y', linestyle='--', alpha=0.7)


# Plot Predicted Behaviors on the second subplot (ax2)
for behavior, y_pos in behavior_mapping.items():
    # Filter data for the current predicted behavior
    behavior_data = processed_df_sorted[(processed_df_sorted['behavior_name'] == behavior) & (processed_df_sorted['data_type'] == 'prediction')]

    if not behavior_data.empty:
        # Plot vertical lines for each global frame index where the predicted behavior occurs
        ax2.vlines(x=behavior_data['global_frame_index'],
                  ymin=y_pos - 0.4,
                  ymax=y_pos + 0.4,
                  color=behavior_colors[behavior],
                  linewidth=1)

# Set y-axis ticks and labels for the predicted behavior plot
ax2.set_yticks(list(behavior_mapping.values()))
ax2.set_yticklabels(list(behavior_mapping.keys()))
ax2.set_ylabel("Predicted Behavior", fontsize=14)
#ax2.set_title("Predicted Behavioral Ethogram Across All Experiment Days", fontsize=16)
ax2.set_ylim(-0.5, len(behavior_mapping) - 0.5)
ax2.grid(axis='y', linestyle='--', alpha=0.7)


# Plot the light cycle on the third subplot (ax3)
# We need to determine the start and end global frame index for each light cycle phase segment
light_cycle_min = processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'])['global_frame_index'].min().reset_index(name='min_global_frame')
light_cycle_max = processed_df_sorted.groupby(['Day', 'Light Cycle Phase', 'original_file'])['global_frame_index'].max().reset_index(name='max_global_frame')

light_cycle_segments = pd.merge(light_cycle_min, light_cycle_max,
                                on=['Day', 'Light Cycle Phase', 'original_file'],
                                how='left').reset_index() # Reset index to make grouped columns regular columns

# Define colors for light and dark cycles
light_cycle_colors = {'Light': 'yellow', 'Dark': 'darkblue'}

# Plot rectangles for each light cycle segment
for index, row in light_cycle_segments.iterrows():
    # Check for NaN values in min/max global frame before plotting
    if pd.notnull(row['min_global_frame']) and pd.notnull(row['max_global_frame']):
        start_frame = row['min_global_frame']
        end_frame = row['max_global_frame']
        light_cycle = row.get('Light Cycle Phase', None) # Use .get() to safely access
        color = light_cycle_colors.get(light_cycle, 'gray') # Default to gray

        ax3.add_patch(plt.Rectangle((start_frame, 0), end_frame - start_frame, 0.5, color=color)) # Decreased height to 0.5


# Set y-axis for the light cycle plot
ax3.set_yticks([0.25]) # Adjust tick position
ax3.set_yticklabels(['Light Cycle'])
ax3.set_ylim(0, 0.5) # Adjust y-axis limit
#ax3.set_xlabel("Global Frame Index", fontsize=14) # Set x-axis label on the lower subplot


# Add vertical lines at day boundaries on all subplots
for index, row in day_boundaries.iterrows():
    ax1.axvline(x=row['global_frame_index'], color='gray', linestyle='--', linewidth=1.5)
    ax2.axvline(x=row['global_frame_index'], color='gray', linestyle='--', linewidth=1.5)
    ax3.axvline(x=row['global_frame_index'], color='gray', linestyle='--', linewidth=1.5)
    # Add day label near the boundary line on the lowest subplot
    ax3.text(row['global_frame_index'], ax3.get_ylim()[0] - 0.5, f"Day {int(row['Day'])}", rotation=45, ha='right', va='top', fontsize=10, color='black') # Adjust y position relative to y-axis limits


# Set the x-axis limits to tightly fit the data for all subplots (shared x-axis)
ax1.set_xlim(processed_df_sorted['global_frame_index'].min(), processed_df_sorted['global_frame_index'].max())

# Adjust space between subplots
plt.subplots_adjust(hspace=0.1) # Adjusted hspace slightly

# Create legends for each plot
# Legend for behaviors (shared for labeled and predicted plots)
legend_handles_behaviors = [mpatches.Patch(color=behavior_colors[behavior], label=behavior) for behavior in behavior_mapping.keys()]
ax1.legend(handles=legend_handles_behaviors, title="Behaviors", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
#ax2.legend(handles=legend_handles_behaviors, title="Behaviors", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)


# Legend for light cycle
legend_handles_lightcycle = [mpatches.Patch(color=color, label=phase) for phase, color in light_cycle_colors.items()]
ax3.legend(handles=legend_handles_lightcycle, title="Light Cycle", bbox_to_anchor=(1.01, 0), loc='lower left', fontsize=12)

ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)

plt.show()
