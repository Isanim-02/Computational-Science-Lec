"""
Interactive Philippines Rainfall Map using Folium
Creates zoomable, clickable HTML maps that open in your browser
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import json

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Folium not installed. Run: pip install folium")


def create_interactive_rainfall_map(lons, lats, predictions, city_names, 
                                   kernel_name='RBF', rmse=None, r2=None,
                                   output_file='philippines_interactive_rainfall.html'):
    """
    Create interactive Folium map with rainfall predictions
    
    Parameters:
    -----------
    lons : array
        Longitude coordinates
    lats : array
        Latitude coordinates
    predictions : array
        Rainfall predictions (mm)
    city_names : array
        City names
    kernel_name : str
        Kernel name for title
    rmse : float, optional
        RMSE value
    r2 : float, optional
        R² value
    output_file : str
        Output HTML filename
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available. Install with: pip install folium")
        return
    
    print(f"\nCreating interactive map: {output_file}")
    
    # Center map on Philippines
    map_center = [12.8797, 121.7740]  # Philippines center
    
    # Create base map
    m = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add different tile layers (user can switch)
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Determine color scale
    min_val = predictions.min()
    max_val = predictions.max()
    
    # Color mapping function
    def get_color(value):
        """Map rainfall value to color"""
        normalized = (value - min_val) / (max_val - min_val)
        
        if normalized < 0.2:
            return '#d73027'  # Red (low)
        elif normalized < 0.4:
            return '#fc8d59'  # Orange
        elif normalized < 0.6:
            return '#fee08b'  # Yellow
        elif normalized < 0.8:
            return '#d9ef8b'  # Light green
        else:
            return '#1a9850'  # Dark green (high)
    
    # Create feature group for markers
    marker_cluster = plugins.MarkerCluster(name='Weather Stations').add_to(m)
    
    # Add markers for each city
    for lon, lat, pred, city in zip(lons, lats, predictions, city_names):
        
        color = get_color(pred)
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 0; color: #2c3e50;">{city}</h4>
            <hr style="margin: 5px 0;">
            <table style="width: 100%; font-size: 13px;">
                <tr>
                    <td><b>Rainfall:</b></td>
                    <td style="text-align: right;">{pred:.2f} mm</td>
                </tr>
                <tr>
                    <td><b>Latitude:</b></td>
                    <td style="text-align: right;">{lat:.4f}°N</td>
                </tr>
                <tr>
                    <td><b>Longitude:</b></td>
                    <td style="text-align: right;">{lon:.4f}°E</td>
                </tr>
                <tr>
                    <td><b>Kernel:</b></td>
                    <td style="text-align: right;">{kernel_name}</td>
                </tr>
            </table>
        </div>
        """
        
        # Add circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city}: {pred:.1f} mm",
            color='black',
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(marker_cluster)
    
    # Add heatmap layer (optional, user can toggle)
    heat_data = [[lat, lon, pred] for lat, lon, pred in zip(lats, lons, predictions)]
    plugins.HeatMap(heat_data, name='Rainfall Heatmap', 
                   min_opacity=0.3, radius=25, blur=30, 
                   gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'yellow', 
                            0.8: 'orange', 1.0: 'red'}).add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 280px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; text-align: center;">
            Philippines Monthly Rainfall
        </h4>
        <p style="margin: 5px 0; font-size: 12px; text-align: center;">
            <b>{kernel_name} Kernel - SVR Model</b>
        </p>
        {f'<p style="margin: 5px 0; font-size: 12px; text-align: center;">RMSE: {rmse:.2f} mm | R²: {r2:.3f}</p>' if rmse and r2 else ''}
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0;"><b>Rainfall (mm/month):</b></p>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background: #d73027; width: 30px; height: 15px; margin-right: 10px;"></div>
            <span>{min_val:.0f} - {min_val + (max_val-min_val)*0.2:.0f} (Low)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background: #fc8d59; width: 30px; height: 15px; margin-right: 10px;"></div>
            <span>{min_val + (max_val-min_val)*0.2:.0f} - {min_val + (max_val-min_val)*0.4:.0f}</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background: #fee08b; width: 30px; height: 15px; margin-right: 10px;"></div>
            <span>{min_val + (max_val-min_val)*0.4:.0f} - {min_val + (max_val-min_val)*0.6:.0f} (Medium)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background: #d9ef8b; width: 30px; height: 15px; margin-right: 10px;"></div>
            <span>{min_val + (max_val-min_val)*0.6:.0f} - {min_val + (max_val-min_val)*0.8:.0f}</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="background: #1a9850; width: 30px; height: 15px; margin-right: 10px;"></div>
            <span>{min_val + (max_val-min_val)*0.8:.0f} - {max_val:.0f} (High)</span>
        </div>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 11px; color: #666;">
            💡 Click markers for details<br>
            🔍 Use layers control (top right)<br>
            🗺️ Zoom/Pan to explore
        </p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add title
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50px; width: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:16px; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h3 style="margin: 0;">🌧️ Philippines Rainfall Prediction - Interactive Map</h3>
        <p style="margin: 5px 0; font-size: 13px;">
            Weather Stations: {len(lons)} | Click markers for details | Toggle layers to explore
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Saved: {output_file}")
    print(f"   Open in browser to interact with the map!")
    
    return m


def create_interactive_comparison_map(predictor, output_file='philippines_kernel_comparison_interactive.html'):
    """
    Create interactive comparison map with all three kernels
    Uses tab control to switch between kernels
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available. Install with: pip install folium")
        return
    
    print(f"\nCreating interactive comparison map...")
    
    # Get data
    lats = predictor.df_monthly['latitude'].values
    lons = predictor.df_monthly['longitude'].values
    city_names = predictor.df_monthly['city_name'].values
    
    # Center map
    map_center = [12.8797, 121.7740]
    
    # Create base map
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
    
    # Create feature groups for each kernel
    for kernel_name, model in predictor.models.items():
        
        # Get predictions
        model.fit(predictor.X_scaled, predictor.y)
        predictions = model.predict(predictor.X_scaled)
        
        # Get metrics
        rmse = predictor.results[kernel_name]['mean_rmse']
        r2 = predictor.results[kernel_name]['mean_r2']
        
        # Create feature group
        fg = folium.FeatureGroup(name=f'{kernel_name} (RMSE: {rmse:.1f}, R²: {r2:.2f})')
        
        # Determine colors
        min_val = predictions.min()
        max_val = predictions.max()
        
        def get_color(value):
            normalized = (value - min_val) / (max_val - min_val)
            if normalized < 0.2: return '#d73027'
            elif normalized < 0.4: return '#fc8d59'
            elif normalized < 0.6: return '#fee08b'
            elif normalized < 0.8: return '#d9ef8b'
            else: return '#1a9850'
        
        # Add markers
        for lon, lat, pred, city in zip(lons, lats, predictions, city_names):
            
            popup_html = f"""
            <div style="font-family: Arial;">
                <h4 style="margin: 0;">{city}</h4>
                <p><b>Kernel:</b> {kernel_name}</p>
                <p><b>Rainfall:</b> {pred:.2f} mm</p>
                <p><b>RMSE:</b> {rmse:.2f} mm</p>
                <p><b>R²:</b> {r2:.3f}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{city}: {pred:.1f} mm",
                color='black',
                fillColor=get_color(pred),
                fillOpacity=0.7,
                weight=1.5
            ).add_to(fg)
        
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h3 style="margin: 0;">🌧️ Kernel Comparison - Interactive</h3>
        <p style="margin: 5px 0; font-size: 13px;">
            Toggle layers to compare different kernels
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    m.save(output_file)
    print(f"Saved: {output_file}")
    print(f"   Toggle layers in top-right to compare kernels!")
    
    return m


def create_interactive_visualizations(predictor):
    """
    Create all interactive Folium maps for a trained predictor
    
    Parameters:
    -----------
    predictor : PhilippinesRainfallPredictorEnhanced
        Trained predictor object
    """
    print("\n" + "="*70)
    print("CREATING INTERACTIVE FOLIUM MAPS")
    print("Zoomable, Clickable HTML Maps")
    print("="*70)
    
    if not FOLIUM_AVAILABLE:
        print("\n❌ Folium not installed!")
        print("Install with: pip install folium")
        print("="*70)
        return
    
    # Get data
    lats = predictor.df_monthly['latitude'].values
    lons = predictor.df_monthly['longitude'].values
    city_names = predictor.df_monthly['city_name'].values
    
    # Create individual maps for each kernel
    for kernel_name, model in predictor.models.items():
        print(f"\nCreating interactive map for {kernel_name} kernel...")
        
        # Train and predict
        model.fit(predictor.X_scaled, predictor.y)
        predictions = model.predict(predictor.X_scaled)
        
        # Get metrics
        rmse = predictor.results[kernel_name]['mean_rmse']
        r2 = predictor.results[kernel_name]['mean_r2']
        
        # Create map
        output_file = f'philippines_interactive_{kernel_name.lower()}.html'
        create_interactive_rainfall_map(
            lons, lats, predictions, city_names,
            kernel_name, rmse, r2, output_file
        )
    
    # Create comparison map
    print(f"\nCreating kernel comparison map...")
    create_interactive_comparison_map(predictor)
    
    print("\n" + "="*70)
    print("INTERACTIVE MAPS COMPLETE!")
    print("="*70)
    print("\nGenerated HTML files:")
    print("  - philippines_interactive_rbf.html")
    print("  - philippines_interactive_polynomial.html")
    print("  - philippines_interactive_sigmoid.html")
    print("  - philippines_kernel_comparison_interactive.html")
    print("\nDouble-click any HTML file to open in your browser!")
    print("   - Zoom and pan to explore")
    print("   - Click markers for details")
    print("   - Toggle layers to compare")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("INTERACTIVE PHILIPPINES MAP MODULE (FOLIUM)")
    print("="*70)
    print("\nThis module creates interactive HTML maps with:")
    print("  - Zoomable, pannable interface")
    print("  - Clickable city markers")
    print("  - Popup details on click")
    print("  - Layer control (switch views)")
    print("  - Heatmap overlay")
    print("  - Fullscreen mode")
    print("\nRequires: pip install folium")
    print("\nUsage: from philippines_interactive_map import create_interactive_visualizations")
    print("="*70)

