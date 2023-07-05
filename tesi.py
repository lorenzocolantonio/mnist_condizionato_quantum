import numpy as np
import matplotlib.pyplot as plt

# Definizione della funzione di Heaviside
def heaviside(x):
    return np.heaviside(x, 0.5)

# Definizione della funzione segno (sign)
def sign(x):
    return np.sign(x)

# Creazione di un array di valori x
x = np.linspace(-1, 1, 100)

# Calcolo dei valori delle funzioni di Heaviside e segno
y_heaviside = heaviside(x)
y_sign = sign(x)

# Creazione del plot con due subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot della funzione di Heaviside
ax1.plot(x, y_heaviside, color='red', label='Heaviside')
ax1.axhline(0, color='black', linestyle='--')
ax1.legend()
ax1.set_title('Heaviside Function')
ax1.set_xlabel('z')
ax1.set_ylabel('y')

# Plot della funzione segno
ax2.plot(x, y_sign, color='red', label='Sign')
ax2.axhline(0, color='black', linestyle='--')
ax2.legend()
ax2.set_title('Sign Function')
ax2.set_xlabel('z')
ax2.set_ylabel('y')

# Visualizzazione del plot
plt.tight_layout()
plt.show()