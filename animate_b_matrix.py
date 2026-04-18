import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

def animate_b_matrix(file_path='sim_history.npz'):
    data = np.load(file_path, allow_pickle=True)
    b_history = data['b_matrix']
    time = data['time']
    
    # Rows and Columns labels
    rows = ['Thrust', 'Roll', 'Pitch', 'Yaw']
    cols = ['Motor 0 (FR)', 'Motor 1 (RR)', 'Motor 2 (RL)', 'Motor 3 (FL)']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap or something clear for effectiveness
    # We'll normalize based on the first (nominal) B matrix
    b_nom = b_history[0]
    
    # Create the initial heatmap with fixed scale to focus on moments
    im = ax.imshow(b_nom, cmap='RdYlGn', interpolation='nearest', vmin=-1e-11, vmax=1e-11)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Effectiveness Magnitude')
    
    # Add labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    
    # Add text annotations for the values
    texts = []
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, f'{b_nom[i, j]:.2e}',
                           ha="center", va="center", color="black", fontsize=8)
            texts.append(text)
            
    ax.set_title(f"B-Matrix Heatmap (Time: {time[0]:.2f}s)")
    fig.tight_layout()

    def update(frame):
        # Downsample to speed up animation generation
        idx = frame * 20 
        if idx >= len(b_history): idx = len(b_history) - 1
            
        b_curr = b_history[idx]
        im.set_data(b_curr)
        
        # Update annotations
        k = 0
        for i in range(len(rows)):
            for j in range(len(cols)):
                texts[k].set_text(f'{b_curr[i, j]:.2e}')
                # Change text color based on background
                # This is a bit complex, let's keep it simple
                k += 1
                
        ax.set_title(f"B-Matrix Heatmap (Time: {time[idx]:.2f}s)")
        return [im] + texts

    num_frames = len(b_history) // 20
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    output_file = 'b_matrix_animation.mp4'
    print(f"Saving animation to {output_file}...")
    try:
        ani.save(output_file, writer='ffmpeg', fps=24)
        print("Success!")
    except Exception as e:
        print(f"Failed to save as mp4: {e}")
        print("Saving as gif instead...")
        ani.save('b_matrix_animation.gif', writer='pillow', fps=24)
        print("Success (GIF)!")

if __name__ == "__main__":
    animate_b_matrix()
