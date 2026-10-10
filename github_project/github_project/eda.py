import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid")
df = pd.read_csv("diamonds.csv")
print(df.shape); print(df.dtypes); print(df.isna().sum().sum())
print(df.describe().round(2).to_string())
print("dups", df.duplicated().sum())
print("zeros", {c:(df[c]==0).sum() for c in "xyz"})
print("outl", df[(df.y>20)|(df.z>20)|(df.table>80)|(df.depth>75)][["carat","depth","table","price","x","y","z"]])
for c in ["cut","color","clarity"]: print(df[c].value_counts().to_dict())
print("skew", df[["carat","price"]].skew().round(2).to_dict(), "logprice skew", np.log(df.price).skew().round(2))
print(df[["carat","depth","table","price","x","y","z"]].corr().round(2).to_string())
print(df.groupby("cut").price.agg(["mean","median","count"]).round(0))
print(df.groupby("clarity").price.agg(["mean","median"]).round(0))
print(df.groupby("color").price.agg(["mean","median"]).round(0))
print(df.groupby("cut").carat.mean().round(3))
# anova cut
groups=[g.price.values for _,g in df.groupby("cut")]
print("anova", stats.f_oneway(*groups))
print("kruskal", stats.kruskal(*groups))
# price per carat
print(df.groupby("cut").apply(lambda g:(g.price/g.carat).median()).round(0))
# clean
c = df[(df.x>0)&(df.y>0)&(df.z>0)&(df.y<20)&(df.z<20)]
print("clean", c.shape, len(df)-len(c))
# figures
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
sns.histplot(df.price,bins=60,ax=ax[0],color="#2b6cb0"); ax[0].set_title("Price (skewed right)")
sns.histplot(np.log10(df.price),bins=60,ax=ax[1],color="#2f855a"); ax[1].set_title("log10(Price)")
plt.tight_layout(); plt.savefig("figures/f1.png",dpi=150); plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
for a,col,o in zip(ax,["cut","color","clarity"],[["Fair","Good","Very Good","Premium","Ideal"],list("DEFGHIJ"),["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]]):
    sns.countplot(data=df,x=col,order=o,ax=a,color="#2b6cb0"); a.set_title(f"Count by {col}"); a.tick_params(axis="x",rotation=30)
plt.tight_layout(); plt.savefig("figures/f2.png",dpi=150); plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.6))
for a,col,o in zip(ax,["cut","color","clarity"],[["Fair","Good","Very Good","Premium","Ideal"],list("DEFGHIJ"),["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]]):
    sns.boxplot(data=df,x=col,y="price",order=o,ax=a,color="#90cdf4",fliersize=1); a.set_title(f"Price by {col}"); a.tick_params(axis="x",rotation=30)
plt.tight_layout(); plt.savefig("figures/f3.png",dpi=150); plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,4))
sns.heatmap(df[["carat","depth","table","price","x","y","z"]].corr(),annot=True,fmt=".2f",cmap="Blues",ax=ax[0],cbar=False); ax[0].set_title("Correlation matrix")
ax[1].scatter(df.carat,df.price,s=1,alpha=.2,color="#2b6cb0"); ax[1].set_xlabel("carat"); ax[1].set_ylabel("price"); ax[1].set_title("Carat vs price")
plt.tight_layout(); plt.savefig("figures/f4.png",dpi=150); plt.close()
