import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

def draw_road_network():
    """
    Read Geneva's edges.shp file and draw the complete road network with different colors for differen
    d types
    """
    # Define file path
    edges_path = Path("paris_center/edges.shp")
    
    # Check if file exists
    if not edges_path.exists():
        print(f"Error: File not found {edges_path}")
        return
    
    # Read edge data
    print("Reading edge data...")
    edges_gdf = gpd.read_file(edges_path)
    print(f"Total edges read: {len(edges_gdf)}")
    
    # 定义道路类型的颜色和宽度（统一宽度）
    road_styles = {
        'motorway': {'color': '#e74c3c', 'width': 0.8, 'label': 'Motorway'},
        'motorway_link': {'color': '#e74c3c', 'width': 0.8, 'label': 'Motorway Link'},
        'trunk': {'color': '#f39c12', 'width': 0.8, 'label': 'Trunk'},
        'trunk_link': {'color': '#f39c12', 'width': 0.8, 'label': 'Trunk Link'},
        'primary': {'color': '#f1c40f', 'width': 0.8, 'label': 'Primary'},
        'primary_link': {'color': '#f1c40f', 'width': 0.8, 'label': 'Primary Link'},
        'secondary': {'color': '#3498db', 'width': 0.8, 'label': 'Secondary'},
        'secondary_link': {'color': '#3498db', 'width': 0.8, 'label': 'Secondary Link'},
        'tertiary': {'color': '#9b59b6', 'width': 0.8, 'label': 'Tertiary'},
        'tertiary_link': {'color': '#9b59b6', 'width': 0.8, 'label': 'Tertiary Link'},
        'residential': {'color': '#95a5a6', 'width': 0.8, 'label': 'Residential'},
        'living_street': {'color': '#7f8c8d', 'width': 0.8, 'label': 'Living Street'},
        'service': {'color': '#bdc3c7', 'width': 0.8, 'label': 'Service'},
        'unclassified': {'color': '#34495e', 'width': 0.8, 'label': 'Unclassified'},
    }
    
    # Draw road network (using geopandas plot for better performance)
    print("Drawing road network...")
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Draw by road type, minor roads first, major roads last (so major roads are on top)
    draw_order = ['service', 'living_street', 'residential', 'unclassified', 
                  'tertiary_link', 'tertiary', 'secondary_link', 'secondary',
                  'primary_link', 'primary', 'trunk_link', 'trunk',
                  'motorway_link', 'motorway']
    
    from matplotlib.lines import Line2D
    legend_handles = []
    
    for highway_type in draw_order:
        # Filter edges by type
        type_edges = edges_gdf[edges_gdf['highway'] == highway_type]
        
        if len(type_edges) == 0:
            continue
        
        style = road_styles.get(highway_type, {'color': 'gray', 'width': 0.8, 'label': highway_type})
        
        print(f"Drawing {highway_type}: {len(type_edges)} edges")
        
        # Use geopandas plot for batch rendering (much faster)
        type_edges.plot(ax=ax, 
                       color=style['color'], 
                       linewidth=style['width'], 
                       alpha=0.7, 
                       zorder=draw_order.index(highway_type))
        
        # Add to legend
        legend_handles.append(Line2D([0], [0], 
                                    color=style['color'], 
                                    linewidth=style['width']*2, 
                                    label=f"{style['label']} ({len(type_edges)})"))
    
    # Add other road types not in draw_order
    other_types = edges_gdf[~edges_gdf['highway'].isin(draw_order)]
    for highway_type in other_types['highway'].unique():
        type_edges = edges_gdf[edges_gdf['highway'] == highway_type]
        style = road_styles.get(highway_type, {'color': 'gray', 'width': 0.8, 'label': highway_type})
        
        print(f"Drawing {highway_type}: {len(type_edges)} edges")
        
        type_edges.plot(ax=ax, 
                       color=style['color'], 
                       linewidth=style['width'], 
                       alpha=0.7)
        
        legend_handles.append(Line2D([0], [0], 
                                    color=style['color'], 
                                    linewidth=style['width']*2, 
                                    label=f"{style['label']} ({len(type_edges)})"))
    
    # Set legend
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.set_title('Geneva Road Network', fontsize=16, pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Display the map in interactive window
    print("\nDisplaying road network map...")
    plt.show()
    
    # Output statistics
    print("\nRoad Network Statistics:")
    print(f"- Total edges: {len(edges_gdf)}")
    print("\nEdges by road type:")
    for highway_type in edges_gdf['highway'].value_counts().index:
        count = len(edges_gdf[edges_gdf['highway'] == highway_type])
        style = road_styles.get(highway_type, {'label': highway_type})
        print(f"  - {style['label']} ({highway_type}): {count} edges")

if __name__ == "__main__":
    draw_road_network()
