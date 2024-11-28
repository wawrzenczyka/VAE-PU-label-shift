# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pi_vec = np.linspace(0.1, 0.9, 9)
pi_wave = np.linspace(0.01, 1, 100)

sns.set_theme()
plt.figure(figsize=(10, 6))

plt.hlines(1, 0, 1, "k")

for pi in pi_vec:
    theta = (1 / pi_wave - 1) / (1 / pi - 1)
    plt.plot(pi_wave, theta)

plt.legend(["Constant $\\theta = 1$"] + [f"$\\pi = {pi:.1f}$" for pi in pi_vec])
plt.ylabel("$\\theta$")
plt.xlabel("$\\tilde{\\pi}$")
plt.ylim(0, 10)

plt.savefig("theta.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

c_vec = np.linspace(0.1, 0.9, 9)

pi = np.linspace(0, 1, 101)
pi_wave = np.linspace(0, 1, 101)

sns.set_theme()
plt.figure(figsize=(10, 9))

for c in c_vec:
    ls_proba = (pi_wave - c * pi_wave) / (1 - c * pi_wave)
    # ls_normal = (pi - c * pi) / (1 - c * pi)
    plt.plot(pi_wave, ls_proba)

plt.legend([f"$c = {c:.1f}$" for c in c_vec])
plt.ylabel("$P(Y = 1 | S = 0)$")
plt.xlabel("$\\pi$")
# plt.ylim(0, 10)

plt.savefig("P_Y1_S0.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %%
