import plotly.express as px
import pandas as pd

# Task data
df = pd.DataFrame([
    dict(Task="Preliminary Research", Start="02-19-2026", Finish="03-12-2026", Team="All"),
    dict(Task="Repository Simulation", Start="02-26-2026", Finish="03-19-2026", Team="All"),
    dict(Task="Component Selection", Start="03-05-2026", Finish="03-26-2026" , Team="Hardware team: Zach, Alfred"),
    dict(Task="Software Design", Start="03-27-2026", Finish="09-10-2026" , Team="Software team: Henry, David"),
    dict(Task="Hardware Design", Start="05-01-2026", Finish="08-14-2026" , Team="Hardware team: Zach, Alfred"),
    dict(Task="GUI Implementation", Start="04-16-2026", Finish="09-10-2026", Team="GUI Team: Jerry, Homoud"),
    dict(Task="Testing for range and doppler", Start="09-11-2026", Finish="11-01-2026" , Team="All"),
    dict(Task="Final Report", Start="11-02-2026", Finish="12-04-2026", Team="All")
])

# Create Gantt chart
fig = px.timeline(
    df,
    x_start="Start", 
    x_end="Finish", 
    y="Task",
    color="Team",
    category_orders={"Task": df["Task"].tolist()}
    )



# Team Assignment
color_discrete_map = {
    "All": "blue",
    "Hardware team: Zach, Alfred": "red",
    "Software team: Henry, David": "green",
    "GUI Team: Jerry, Homoud": "orange"
    },

# layout
fig.update_layout(
    font=dict(family="Arial", size=18, color="black")
    )

fig.show()
