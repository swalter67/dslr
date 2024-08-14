import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('../datasets/dataset_train.csv')

# List of courses
courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

# Create a subplot figure with 13 rows
fig = make_subplots(rows=len(courses), cols=1, subplot_titles=courses)

# Plot histograms for each course
for i, course in enumerate(courses):
    if course in df.columns:
        # Filter out NaN values and plot data for each Hogwarts House
        for house in df['Hogwarts House'].dropna().unique():
            course_data = df[(df['Hogwarts House'] == house) & df[course].notna()][course]
            if not course_data.empty:
                fig.add_trace(go.Histogram(x=course_data, name=house, opacity=0.5, bingroup=i),
                              row=i + 1, col=1)

# Update the layout for a cleaner look
fig.update_layout(
    height=2000,  # Adjust height based on the number of subplots
    width=800,
    title_text="Histograms of Courses by Hogwarts House",
    showlegend=True,
    legend_title_text='Houses'
)
fig.update_xaxes(title_text="Scores")
fig.update_yaxes(title_text="Density")

# Set binning mode for all histograms
fig.update_traces(overwrite=True, histnorm='probability density', autobinx=True)

# Show the plot
fig.show()


# Which Hogwarts course has a homogeneous score distribution between the four houses ?