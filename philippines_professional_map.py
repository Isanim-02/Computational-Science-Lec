"""
Professional Philippines Map Visualization
Enhanced version with regional boundaries similar to official maps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False


# Philippines Regional Boundaries (Simplified)
PHILIPPINES_REGIONS = {
    'Luzon': {
        'Ilocos': [(120.3, 18.0), (120.5, 17.0), (120.7, 16.5), (120.4, 16.0), (120.2, 16.5), (120.3, 18.0)],
        'Cagayan Valley': [(121.5, 18.5), (122.3, 17.5), (122.0, 16.5), (121.3, 17.0), (121.5, 18.5)],
        'Central Luzon': [(120.5, 16.0), (121.2, 15.8), (121.5, 15.0), (121.0, 14.5), (120.3, 15.0), (120.5, 16.0)],
        'NCR': [(120.9, 14.7), (121.2, 14.7), (121.2, 14.4), (120.9, 14.4), (120.9, 14.7)],
        'CALABARZON': [(120.8, 14.5), (122.0, 14.3), (122.2, 13.8), (121.8, 13.5), (121.0, 13.8), (120.8, 14.5)],
        'MIMAROPA': [(119.5, 13.5), (120.5, 13.0), (121.0, 12.5), (121.0, 11.8), (120.0, 12.0), (119.3, 12.8), (119.5, 13.5)],
        'Bicol': [(123.0, 14.0), (124.2, 13.5), (124.5, 12.8), (124.0, 12.3), (123.2, 12.5), (122.8, 13.2), (123.0, 14.0)],
    },
    'Visayas': {
        'Western Visayas': [(121.8, 11.8), (123.0, 11.5), (123.2, 10.8), (122.5, 10.3), (121.8, 10.8), (121.8, 11.8)],
        'Central Visayas': [(123.2, 11.2), (124.5, 10.8), (124.7, 10.0), (124.3, 9.5), (123.5, 9.8), (123.0, 10.5), (123.2, 11.2)],
        'Eastern Visayas': [(124.5, 12.5), (125.5, 12.0), (125.7, 11.0), (125.3, 10.5), (124.5, 11.0), (124.5, 12.5)],
    },
    'Mindanao': {
        'Zamboanga': [(121.8, 9.0), (123.0, 8.5), (123.2, 7.5), (122.5, 6.8), (122.0, 7.5), (121.8, 9.0)],
        'Northern Mindanao': [(123.5, 9.5), (125.3, 9.0), (125.5, 8.2), (124.5, 8.0), (123.8, 8.5), (123.5, 9.5)],
        'Davao': [(125.0, 8.0), (126.5, 7.5), (126.8, 6.5), (126.3, 5.8), (125.5, 6.2), (124.8, 7.0), (125.0, 8.0)],
        'SOCCSKSARGEN': [(124.0, 7.5), (125.2, 7.2), (125.5, 6.5), (125.0, 5.9), (124.0, 6.3), (123.5, 7.0), (124.0, 7.5)],
        'Caraga': [(125.5, 10.0), (126.5, 9.5), (126.8, 8.5), (126.3, 8.0), (125.5, 8.5), (125.5, 10.0)],
        'BARMM': [(121.5, 7.5), (122.5, 7.0), (122.8, 6.0), (122.0, 5.5), (121.2, 6.0), (121.0, 7.0), (121.5, 7.5)],
    }
}

# Region colors (professional palette)
REGION_COLORS = {
    'Ilocos': '#E74C3C',           # Red
    'Cagayan Valley': '#3498DB',    # Blue
    'Central Luzon': '#9B59B6',     # Purple
    'NCR': '#F39C12',               # Orange
    'CALABARZON': '#1ABC9C',        # Turquoise
    'MIMAROPA': '#E67E22',          # Dark Orange
    'Bicol': '#2ECC71',             # Green
    'Western Visayas': '#34495E',   # Dark Gray
    'Central Visayas': '#16A085',   # Dark Turquoise
    'Eastern Visayas': '#27AE60',   # Dark Green
    'Zamboanga': '#8E44AD',         # Dark Purple
    'Northern Mindanao': '#2980B9', # Dark Blue
    'Davao': '#C0392B',             # Dark Red
    'SOCCSKSARGEN': '#D35400',      # Burnt Orange
    'Caraga': '#7F8C8D',            # Gray
    'BARMM': '#F1C40F',             # Yellow
}


def plot_philippines_regions(ax, alpha=0.15, show_labels=False):
    """
    Plot Philippines regional boundaries on an axis
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    alpha : float
        Transparency of region fills
    show_labels : bool
        Whether to show region labels
    """
    for island, regions in PHILIPPINES_REGIONS.items():
        for region_name, coords in regions.items():
            coords_array = np.array(coords)
            
            # Get color for this region
            color = REGION_COLORS.get(region_name, '#95A5A6')
            
            # Draw filled region
            polygon = Polygon(coords_array, closed=True, 
                            facecolor=color, edgecolor='darkblue',
                            alpha=alpha, linewidth=1.5, zorder=1)
            ax.add_patch(polygon)
            
            # Draw boundary
            ax.plot(coords_array[:, 0], coords_array[:, 1],
                   color='darkblue', linewidth=2, alpha=0.7, zorder=2)
            
            # Add region label
            if show_labels:
                center_lon = coords_array[:, 0].mean()
                center_lat = coords_array[:, 1].mean()
                ax.text(center_lon, center_lat, region_name,
                       fontsize=7, ha='center', va='center',
                       fontweight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.7),
                       zorder=5)


def create_professional_rainfall_map(lons, lats, predictions, kernel_name='RBF',
                                    rmse=None, r2=None, output_file=None):
    """
    Create professional rainfall map with regional boundaries
    
    Parameters:
    -----------
    lons : array
        Longitude coordinates
    lats : array  
        Latitude coordinates
    predictions : array
        Rainfall predictions (mm)
    kernel_name : str
        Kernel name for title
    rmse : float, optional
        RMSE value to display
    r2 : float, optional
        R² value to display
    output_file : str, optional
        Output filename
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 16))
    
    # Plot regional boundaries first (as base layer)
    plot_philippines_regions(ax, alpha=0.12, show_labels=True)
    
    # Create scatter plot of predictions
    scatter = ax.scatter(lons, lats, c=predictions, 
                        s=200, cmap='RdYlGn', alpha=0.85,
                        edgecolors='black', linewidth=1.2,
                        vmin=predictions.min(), vmax=predictions.max(),
                        zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Monthly Rainfall (mm)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Major cities
    major_cities = {
        'Manila': (120.98, 14.60),
        'Quezon City': (121.03, 14.63),
        'Cebu City': (123.89, 10.32),
        'Davao City': (125.61, 7.07),
        'Baguio': (120.59, 16.42),
        'Iloilo City': (122.57, 10.72),
        'Cagayan de Oro': (124.65, 8.48),
        'General Santos': (125.17, 6.11),
        'Zamboanga': (122.07, 6.91),
        'Bacolod': (122.84, 10.68)
    }
    
    for city, (lon, lat) in major_cities.items():
        # Check if within plot bounds
        if 116 <= lon <= 127 and 4.5 <= lat <= 19.5:
            ax.plot(lon, lat, marker='*', color='red', 
                   markersize=15, markeredgecolor='darkred',
                   markeredgewidth=1.5, zorder=6)
            
            # Label with white background for readability
            ax.annotate(city, xy=(lon, lat), xytext=(5, 5),
                       textcoords='offset points', fontsize=9,
                       fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white', 
                                edgecolor='darkred',
                                alpha=0.9, linewidth=1.5),
                       zorder=7)
    
    # Styling
    ax.set_xlabel('Longitude (°E)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    
    # Title with performance metrics
    title = f'Philippines Monthly Rainfall Prediction\n{kernel_name} Kernel - Support Vector Regression'
    if rmse is not None and r2 is not None:
        title += f'\nRMSE: {rmse:.2f} mm | R²: {r2:.3f}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Grid and limits
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlim(116, 127)
    ax.set_ylim(4.5, 19.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add compass rose
    add_compass_rose(ax)
    
    # Add legend for regions
    add_region_legend(ax)
    
    # Add text box with info
    info_text = f'Weather Stations: {len(lons)}\n'
    info_text += f'Regional Boundaries Shown\n'
    info_text += f'Data: 2020-2023'
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white', alpha=0.85,
                    edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_file}")
    
    plt.close()


def add_compass_rose(ax):
    """Add a compass rose to the map"""
    # Position in axis coordinates
    x, y = 0.95, 0.08
    size = 0.03
    
    # North arrow
    ax.annotate('', xy=(x, y + size), xytext=(x, y),
               xycoords='axes fraction',
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(x, y + size + 0.01, 'N', transform=ax.transAxes,
           ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # E-W markers
    ax.text(x + size*0.7, y, 'E', transform=ax.transAxes,
           ha='left', va='center', fontsize=11, fontweight='bold')
    ax.text(x - size*0.7, y, 'W', transform=ax.transAxes,
           ha='right', va='center', fontsize=11, fontweight='bold')


def add_region_legend(ax):
    """Add legend showing major regions"""
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='Luzon Regions', alpha=0.7),
        mpatches.Patch(facecolor='#16A085', label='Visayas Regions', alpha=0.7),
        mpatches.Patch(facecolor='#2980B9', label='Mindanao Regions', alpha=0.7),
        mpatches.Line2D([0], [0], marker='*', color='w', 
                       markerfacecolor='red', markersize=12,
                       label='Major Cities', markeredgecolor='darkred')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left',
             fontsize=10, framealpha=0.9, edgecolor='black', 
             fancybox=True, shadow=True)


def create_kriging_professional_map(lons, lats, predictions, kernel_name='RBF',
                                   rmse=None, r2=None, n_grid=60):
    """
    Create Kriging interpolation map with professional styling
    """
    if not KRIGING_AVAILABLE:
        print("PyKrige not available. Creating scatter map...")
        create_professional_rainfall_map(lons, lats, predictions, kernel_name,
                                        rmse, r2, 
                                        f'philippines_professional_{kernel_name.lower()}.png')
        return
    
    print(f"\nCreating professional Kriging map for {kernel_name} kernel...")
    
    # Kriging interpolation
    lat_min, lat_max = 4.5, 19.5
    lon_min, lon_max = 116.0, 127.0
    
    grid_lat = np.linspace(lat_min, lat_max, n_grid)
    grid_lon = np.linspace(lon_min, lon_max, n_grid)
    
    try:
        OK = OrdinaryKriging(lons, lats, predictions,
                           variogram_model='spherical',
                           verbose=False, enable_plotting=False)
        
        z, ss = OK.execute('grid', grid_lon, grid_lat)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        # ===== LEFT: Rainfall Prediction =====
        ax1 = axes[0]
        
        # Plot regional boundaries
        plot_philippines_regions(ax1, alpha=0.12, show_labels=True)
        
        # Contour plot
        contour = ax1.contourf(grid_lon, grid_lat, z, levels=20,
                              cmap='RdYlGn', alpha=0.7, zorder=1)
        
        # Add stations
        scatter1 = ax1.scatter(lons, lats, c=predictions, s=100,
                             cmap='RdYlGn', edgecolors='black',
                             linewidth=1.5, alpha=0.95, zorder=4)
        
        # Major cities
        cities = {'Manila': (120.98, 14.60), 'Cebu': (123.89, 10.32),
                 'Davao': (125.61, 7.07), 'Baguio': (120.59, 16.42)}
        for city, (lon, lat) in cities.items():
            ax1.plot(lon, lat, '*', color='red', markersize=15,
                    markeredgecolor='darkred', markeredgewidth=1.5, zorder=6)
            ax1.annotate(city, xy=(lon, lat), xytext=(5, 5),
                        textcoords='offset points', fontsize=10,
                        fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='white', alpha=0.9,
                                 edgecolor='darkred', linewidth=1.5),
                        zorder=7)
        
        ax1.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
        
        title1 = f'Rainfall Prediction (Kriging) - {kernel_name} Kernel'
        if rmse is not None and r2 is not None:
            title1 += f'\nRMSE: {rmse:.2f} mm | R²: {r2:.3f}'
        ax1.set_title(title1, fontsize=14, fontweight='bold')
        
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(116, 127)
        ax1.set_ylim(4.5, 19.5)
        ax1.set_aspect('equal')
        
        cbar1 = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Rainfall (mm)', fontsize=12, fontweight='bold')
        
        add_compass_rose(ax1)
        
        # ===== RIGHT: Uncertainty Map =====
        ax2 = axes[1]
        
        # Plot regional boundaries
        plot_philippines_regions(ax2, alpha=0.08, show_labels=False)
        
        # Variance contour
        contour2 = ax2.contourf(grid_lon, grid_lat, ss, levels=20,
                               cmap='YlOrRd', alpha=0.8, zorder=1)
        
        # Stations (low uncertainty)
        ax2.scatter(lons, lats, s=100, c='blue', alpha=0.6,
                   edgecolors='darkblue', linewidth=1.5, zorder=4,
                   label='Weather Stations')
        
        ax2.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
        ax2.set_title('Prediction Uncertainty (Kriging Variance)\n' +
                     'Red = High Uncertainty (Need More Stations)',
                     fontsize=14, fontweight='bold')
        
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(116, 127)
        ax2.set_ylim(4.5, 19.5)
        ax2.set_aspect('equal')
        
        cbar2 = plt.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Variance', fontsize=12, fontweight='bold')
        
        ax2.legend(loc='lower left', fontsize=11, framealpha=0.9)
        
        add_compass_rose(ax2)
        
        plt.suptitle('Philippines Monthly Rainfall Analysis\n' +
                    'Support Vector Regression with Regional Context',
                    fontsize=17, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        filename = f'philippines_professional_kriging_{kernel_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"Kriging failed: {e}")
        create_professional_rainfall_map(lons, lats, predictions, kernel_name,
                                        rmse, r2,
                                        f'philippines_professional_{kernel_name.lower()}.png')


def create_professional_visualizations(predictor):
    """
    Create all professional map visualizations for a trained predictor
    
    Parameters:
    -----------
    predictor : PhilippinesRainfallPredictorEnhanced
        Trained predictor object
    """
    print("\n" + "="*70)
    print("CREATING PROFESSIONAL PHILIPPINES MAPS")
    print("With Regional Boundaries & Enhanced Styling")
    print("="*70)
    
    lats = predictor.df_monthly['latitude'].values
    lons = predictor.df_monthly['longitude'].values
    
    # Create maps for each kernel
    for kernel_name, model in predictor.models.items():
        print(f"\nProcessing {kernel_name} kernel...")
        
        # Train and predict
        model.fit(predictor.X_scaled, predictor.y)
        predictions = model.predict(predictor.X_scaled)
        
        # Get performance metrics
        rmse = predictor.results[kernel_name]['mean_rmse']
        r2 = predictor.results[kernel_name]['mean_r2']
        
        # Create Kriging map
        create_kriging_professional_map(lons, lats, predictions, kernel_name,
                                       rmse, r2, n_grid=60)
    
    print("\n" + "="*70)
    print("✅ PROFESSIONAL MAPS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - philippines_professional_kriging_rbf.png")
    print("  - philippines_professional_kriging_polynomial.png")
    print("  - philippines_professional_kriging_sigmoid.png")
    print("\n✨ Maps include regional boundaries & professional styling!")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PROFESSIONAL PHILIPPINES MAP MODULE")
    print("="*70)
    print("\nThis module creates professional maps with:")
    print("  ✓ Regional boundaries (Luzon, Visayas, Mindanao)")
    print("  ✓ Major cities labeled")
    print("  ✓ Compass rose")
    print("  ✓ Professional color schemes")
    print("  ✓ Enhanced styling")
    print("\nUsage: from philippines_professional_map import create_professional_visualizations")
    print("="*70)

