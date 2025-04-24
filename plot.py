import os
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots



def create_rgb_colormap(points):
    """
    Create a colormap for the points based on their coordinates.
    The color is determined by the distance from the origin.
    """
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    colors_hex = [mcolors.rgb2hex(c) for c in points]
    
    return colors_hex 


def plot_target(filename, points, plots_path, show=False):
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue'
        )
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=filename,
        width=800,
        height=800
    )
    if show: fig.show()
    
    fig.write_html(f'{plots_path}/{filename}.html')
    fig.write_image(f'{plots_path}/{filename}.png')


def start_end_subplot(x_0, x_1_estimated, run_name='Title', plots_path='./', show=False):
    x_0 = x_0.cpu().numpy()
    x_1_estimated = x_1_estimated#.cpu().numpy()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=("Source (x_0)", "Estimated (x_T)"))
    colors_hex = create_rgb_colormap(x_0)

    # Plot the initial point cloud
    fig.add_trace(go.Scatter3d(
        x=x_0[:, 0],
        y=x_0[:, 1],
        z=x_0[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Corresponding colors for each point
        ),
        name='Initial Points'
    ), row=1, col=1)

    # Plot the transformed point cloud (from flow[-1])
    fig.add_trace(go.Scatter3d(
        x=x_1_estimated[:, 0],
        y=x_1_estimated[:, 1],
        z=x_1_estimated[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Same colors to encode correspondence
        ),
        name='Transformed Points'
    ), row=1, col=2)


    # Adjust layout parameters to make the figure bigger
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=run_name,
        width=1800,
        height=800
    )
    
    if show: fig.show()
       
    fig.write_html(f'{plots_path}/{run_name}.html')
    fig.write_image(f'{plots_path}/{run_name}.png')
    
    
def plot_points(points, run_name, plots_path, title, show=False):
    points = points.cpu().numpy()
    
    colors_hex = create_rgb_colormap(points)
    fig = go.Figure()

    # Plot the point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # corresponding colors for each point
        ),
        name='Points'
    ))

    # Adjust layout parameters
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=f"{run_name} - {title}",
        width=900,  # Half the width since we're only showing one plot
        height=800
    )
    
    if show: 
        fig.show()
       
    fig.write_image(f'{plots_path}/{run_name}.png')

   
def start_end_subplot_volume(x_0, x_1_estimated, run_name, plots_path, show=False):
    x_0 = x_0.cpu().numpy()
    x_1_estimated = x_1_estimated.cpu().numpy()
    
    # Compute distances from the center for x_0
    distances = ((x_0 ** 2).sum(axis=1)) ** 0.5
    norm = mcolors.Normalize(vmin=distances.min(), vmax=distances.max())
    colormap = cm.get_cmap('viridis')
    colors_hex = [mcolors.rgb2hex(colormap(norm(d))) for d in distances]

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]], subplot_titles=("Source (x_0)", "Estimated (x_T)"))

    # Plot the initial point cloud
    fig.add_trace(go.Scatter3d(
        x=x_0[:, 0],
        y=x_0[:, 1],
        z=x_0[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Corresponding colors for each point
        ),
        name='Initial Points'
    ), row=1, col=1)

    # Plot the transformed point cloud (from flow[-1])
    fig.add_trace(go.Scatter3d(
        x=x_1_estimated[:, 0],
        y=x_1_estimated[:, 1],
        z=x_1_estimated[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_hex,  # Same colors to encode correspondence
        ),
        name='Transformed Points'
    ), row=1, col=2)

    # Adjust layout parameters to make the figure bigger
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        scene2=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=run_name,
        width=1800,
        height=800
    )
    
    if show: 
        fig.show()
       
    fig.write_html(f'{plots_path}/{run_name}_radius.html')
    fig.write_image(f'{plots_path}/{run_name}_radius.png')