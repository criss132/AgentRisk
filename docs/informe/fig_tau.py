import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False})
tau=[.15,.175,.2,.225,.25,.275,.3,.325,.35,.375,.4,.425,.45,.475,.5,.525,.55,.575,.6,.625,.65,.675,.7,.725,.75,.775,.8,.825,.85]
mae=[.4278,.4243,.4243,.4102,.412,.4102,.4102,.4137,.4155,.4208,.4173,.4014,.3873,.3838,.3961,.3803,.3891,.4049,.4085,.4173,.4137,.4225,.4331,.4331,.4401,.4401,.4437,.4437,.4454]
sub=[6.99,8.39,9.79,9.79,9.79,9.79,10.49,10.49,11.89,12.59,13.29,13.29,14.69,15.38,16.08,16.08,16.08,16.78,16.78,16.78,16.78,16.78,16.78,16.78,16.78,16.78,16.78,16.78,16.78]
fig,ax=plt.subplots(figsize=(3.4,2.2))
ax.plot(tau,sub,color="#C44E52",lw=1.5,label="Subestimación grave (%)")
ax.axhline(10,color="#C44E52",ls=":",lw=0.8)
ax.set_xlabel(r"Umbral $\tau_{S1}$"); ax.set_ylabel("Subestimación grave (%)",color="#C44E52")
ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(tau,mae,color="#4C72B0",lw=1.5); ax2.set_ylabel("MAE S1",color="#4C72B0"); ax2.set_ylim(0.36,0.46)
ax.axvline(0.225,color="gray",ls="--",lw=0.8); ax.text(0.235,16.2,r"$\tau$=0,225",fontsize=7,color="gray")
ax.text(0.5,10.4,"límite 10 %",fontsize=7,color="#C44E52")
fig.tight_layout(); fig.savefig("fig/tau_sweep.pdf")
