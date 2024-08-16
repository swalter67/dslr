import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    try:
        # Load the data
        df = pd.read_csv('../datasets/dataset_train.csv')

        # List of courses
        courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
                'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

        # Color map for Hogwarts Houses
        house_colors = {
            "Ravenclaw": "blue",      # Blue for Ravenclaw
            "Slytherin": "green",     # Green for Slytherin
            "Gryffindor": "red",      # Red for Gryffindor
            "Hufflepuff": "yellow"    # Yellow for Hufflepuff
        }

        # Create a subplot figure with one row per course
        fig = make_subplots(rows=len(courses), cols=1, subplot_titles=courses)

        # Plot scatter plots for each course
        for i, course in enumerate(courses):
            if course in df.columns:
                # Filter out NaN values and plot data for each Hogwarts House
                for house in df['Hogwarts House'].dropna().unique():
                    course_data = df[(df['Hogwarts House'] == house) & df[course].notna()]
                    if not course_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=course_data.index, 
                                y=course_data[course], 
                                mode='markers',
                                name=house,
                                marker=dict(color=house_colors[house])  # Use predefined house color
                            ),
                            row=i + 1, col=1
                        )

        # Update the layout for a cleaner look
        fig.update_layout(
            height=5000,  # Adjust height based on the number of subplots
            width=800,
            title_text="Scatter Plots of Courses by Hogwarts House",
            showlegend=True,
            legend_title_text='Houses'
        )
        fig.update_xaxes(title_text="Student Index")
        fig.update_yaxes(title_text="Scores")

        # Show the plot
        fig.show()


        # history of magic et transfiguration meme repartition



        # Arithmancy et  Care of Magical Creatures sont idem sur une repartion homogene des 4 maisons

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
