try:
    import pandas as pd
    import plotly.express as px
except ImportError:
    print("Necessary libraries are not installed. \
          Please run `pip install -r requirements.txt`")


def main():
    try:
        # Load the data
        df = pd.read_csv('../datasets/dataset_train.csv')

        # List of courses, assumed to be the features of interest
        courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
                   'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
                   'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

        # Filter the DataFrame to include only the relevant courses
        df_filtered = df[courses + ['Hogwarts House']].dropna()  # Including house for coloring

        # Create the scatter plot matrix
        fig = px.scatter_matrix(df_filtered,
                                dimensions=courses,
                                color='Hogwarts House',
                                title='Scatter Plot Matrix of Courses',
                                labels={col: col.replace('_', ' ') for col in courses},  # Optional label formatting
                                height=1700, width=1500)

        # Show the plot
        fig.show()

        # *Arithmancy*, *Potions* and *Care of Magical Creatures*     repartition homogene
        # *Divination*, *Muggle Studies*, *History of Magic*, *Transfiguration*, *Charms* and *Flying* pour une regression lineaire

    except Exception as e:
        print(f'An error has occurred: { e }')


if __name__ == "__main__":
    main()
