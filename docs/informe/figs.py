import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
# Fig: S1 confusion
M=np.array([[64,1,0,1,0],[5,3,0,0,0],[2,2,0,26,2],[0,4,0,29,2],[0,0,0,2,0]])
fig,ax=plt.subplots(figsize=(3.4,2.9))
im=ax.imshow(M,cmap="Blues")
for i in range(5):
    for j in range(5):
        ax.text(j,i,M[i,j],ha="center",va="center",color="white" if M[i,j]>35 else "black",fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(range(1,6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1,6))
ax.set_xlabel("Severidad predicha (M1)"); ax.set_ylabel("Severidad real")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("fig/conf_s1.pdf"); plt.close()
# Fig: S3 F1 macro
fig,ax=plt.subplots(figsize=(3.4,2.2))
lab=["LLM un\nprompt","Siempre\npermitir","Monitor\n(M1+política)"]
A=[0.139,0.225,0.534]; B=[0.176,0.211,0.634]
x=np.arange(3); w=0.36
b1=ax.bar(x-w/2,A,w,label="Política A",color="#4C72B0"); b2=ax.bar(x+w/2,B,w,label="Política B",color="#DD8452")
for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.2f}",ha="center",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("F1 macro (dev, 143)"); ax.set_ylim(0,0.75); ax.legend(frameon=False,fontsize=7)
fig.tight_layout(); fig.savefig("fig/s3_f1.pdf"); plt.close()
# Fig: harm vs safe subgroup & J2a by domain
fig,ax=plt.subplots(figsize=(3.4,2.2))
dom=["Privacy\n(45)","Security\n(16)","Financial\n(13)","Operat.\n(12)","Physical\n(12)"]
v=[.778,.188,.192,.083,.125]
bars=ax.bar(dom,v,color=["#4C72B0"]+["#C44E52"]*4)
for b,val in zip(bars,v): ax.text(b.get_x()+b.get_width()/2,val+0.015,f"{val:.2f}",ha="center",fontsize=7)
ax.set_ylabel("Jaccard S2a por dominio"); ax.set_ylim(0,0.95)
fig.tight_layout(); fig.savefig("fig/j2a_dominio.pdf"); plt.close()
