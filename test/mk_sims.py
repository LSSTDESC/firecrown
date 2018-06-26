import numpy as np
import scipy.linalg as la
import pyccl as ccl
import matplotlib.pyplot as plt
import copy
import sacc
from scipy.interpolate import interp1d

## settings
is_simple=True
if is_simple :
    dz_bias=0.05
    bias_g=1
    bias_l=0
    fname_cl="sims/c_ell_mean_phz.sacc"
    fname_xi="sims/xi_th_mean_phz.sacc"
else :
    dz_bias=0.05
    bias_g=np.pi
    bias_l=np.pi*1E-2
    fname_cl="sims/c_ell_mean.sacc"
    fname_xi="sims/xi_th_mean.sacc"
zbins=[0.7,0.9,1.1]
ndens_bins=[4.,4.,4.]
zbin_size=0.1
lmin_use=10
lmax_use=1000
n_ell=20
tmin_use=3.0
tmax_use=300.0
n_theta=20
fsky=0.4
nth_extra=10
nsims=1000

ell_edges=np.logspace(np.log10(lmin_use),np.log10(lmax_use),n_ell+1).astype(int)
ell_means=0.5*(ell_edges[1:]+ell_edges[:-1])
ell_diff=(ell_edges[1:]-ell_edges[:-1])
th_edges=np.logspace(np.log10(tmin_use),np.log10(tmax_use),n_theta+1)
th_means=0.5*(th_edges[1:]+th_edges[:-1])
th_diff=(th_edges[1:]-th_edges[:-1])

cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.67,sigma8=0.8,n_s=0.96)

#Generate tracers
sacc_t_list=[]
ccl_t_list=[]
tracer_types=[]
noises=[]
for i,z in enumerate(zbins):
    zar=np.arange(z-3*zbin_size,z+3*zbin_size,0.001)
    bias=bias_g*np.ones_like(zar)
    Nz=np.exp(-(z-zar)**2/(2*zbin_size**2))
    T=sacc.Tracer("lenses_"+str(i),"spin0",zar,Nz,exp_sample="lenses")
    T.addColumns({'b':bias})
    sacc_t_list.append(T)
    ccl_t_list.append(ccl.ClTracerNumberCounts(cosmo,False,False,(zar+dz_bias,Nz),(zar+dz_bias,bias)))
    tracer_types.append('S')
    noises.append((np.pi/180/60)**2/ndens_bins[i])
for i,z in enumerate(zbins):
    zar=np.arange(z-3*zbin_size,z+3*zbin_size,0.001)+0.05
    bias_ia=bias_l*np.ones_like(zar)
    fred=np.ones_like(zar)
    Nz=np.exp(-(z-zar)**2/(2*zbin_size**2))
    T=sacc.Tracer("sources_"+str(i),"spin2",zar,Nz,exp_sample="sources")
    sacc_t_list.append(T)
    ccl_t_list.append(ccl.ClTracerLensing(cosmo,True,(zar+dz_bias,Nz),bias_ia=(zar+dz_bias,bias_ia),f_red=(zar+dz_bias,fred)))
    tracer_types.append('E')
    noises.append((0.26*np.pi/180/60)**2/ndens_bins[i])

#Generate base Cls
ntracers=len(ccl_t_list)
nxcorr=(ntracers*(ntracers+1))/2
larr_sample=np.concatenate((np.linspace(2,49,48),np.logspace(np.log10(50.),np.log10(6E4),200)))
larr_full=np.arange(6E4)
cls_full=np.zeros([nxcorr,len(larr_full)])
nls_full=np.zeros([nxcorr,len(larr_full)])
ixcorr=0
twod_order={}
for i1 in np.arange(ntracers) :
    for i2 in np.arange(i1,ntracers) :
        print i1,i2
        cls_sample=ccl.angular_cl(cosmo,ccl_t_list[i1],ccl_t_list[i2],larr_sample)
        clf=interp1d(larr_sample,cls_sample,bounds_error=False,fill_value=0)
        cls_full[ixcorr,:]=clf(larr_full)
        if i1==i2 :
            nls_full[ixcorr,:]=noises[i1]
        twod_order[(i1,i2)]=ixcorr
        twod_order[(i2,i1)]=ixcorr
        ixcorr+=1

#Generate full covariance matrix
covar_full=np.zeros([len(larr_full),nxcorr,nxcorr])
covar_binned=np.zeros([nxcorr*n_ell,nxcorr*n_ell])
ixcorr1=0
for a in np.arange(ntracers):
    for b in np.arange(a,ntracers):
        ixcorr2=0
        for c in np.arange(ntracers):
            for d in np.arange(c,ntracers):
                c_ac=cls_full[twod_order[(a,c)],:]+nls_full[twod_order[(a,c)],:]
                c_bd=cls_full[twod_order[(b,d)],:]+nls_full[twod_order[(b,d)],:]
                c_ad=cls_full[twod_order[(a,d)],:]+nls_full[twod_order[(a,d)],:]
                c_bc=cls_full[twod_order[(b,c)],:]+nls_full[twod_order[(b,c)],:]
                covar_full[:,ixcorr1,ixcorr2]=(c_ac*c_bd+c_ad*c_bc)/(fsky*(2*larr_full+1.))
                l_indices=ell_means.astype(int)
                c_ac=cls_full[twod_order[(a,c)],l_indices]+nls_full[twod_order[(a,c)],l_indices]
                c_bd=cls_full[twod_order[(b,d)],l_indices]+nls_full[twod_order[(b,d)],l_indices]
                c_ad=cls_full[twod_order[(a,d)],l_indices]+nls_full[twod_order[(a,d)],l_indices]
                c_bc=cls_full[twod_order[(b,c)],l_indices]+nls_full[twod_order[(b,c)],l_indices]
                covar_binned[ixcorr1*n_ell:(ixcorr1+1)*n_ell,ixcorr2*n_ell:(ixcorr2+1)*n_ell]=np.diag((c_ac*c_bd+c_ad*c_bc)/(fsky*(2*ell_means+1.)*ell_diff))
                ixcorr2+=1
        ixcorr1+=1
cholesky_full=np.linalg.cholesky(covar_full)

def get_binned_cl(clfl) :
    #return clfl[:,ell_means.astype(int)]
    return np.transpose(np.array([np.mean(clfl[:,ell_edges[i]:ell_edges[i+1]],axis=1) for i in np.arange(n_ell)]))

def get_cl_sim(clfl) :
    return clfl+np.transpose(np.sum(cholesky_full[:,:,:]*np.random.randn(len(larr_full),nxcorr)[:,None,:],axis=2))

def get_xi_one(cl,corr_code) :
    th_full=(th_edges[:-1,None]+(th_edges[1:,None]-th_edges[:-1,None])*(np.arange(nth_extra)/float(nth_extra))[None,:]).flatten()
    xi_full=ccl.correlation(cosmo,larr_full,cl,th_full/60.,corr_code)
    return np.mean(xi_full.reshape([n_theta,nth_extra]),axis=1)

def get_xi_all(cl_all) :
    xi_all=[]
    for i1 in np.arange(ntracers) :
        sc1= (sacc_t_list[i1].type=='spin0')
        for i2 in np.arange(i1,ntracers) :
            sc2= (sacc_t_list[i2].type=='spin0')
            ixcorr=twod_order[(i1,i2)]
            if sc1 and sc2 :
                xi_all.append(get_xi_one(cl_all[ixcorr],'gg'))
            elif sc1 or sc2 :
                xi_all.append(get_xi_one(cl_all[ixcorr],'gl'))
            elif (not sc1) and (not sc2) : #2-2
                xi_all.append(get_xi_one(cl_all[ixcorr],'l+'))  
                xi_all.append(get_xi_one(cl_all[ixcorr],'l-'))
    return np.array(xi_all)

cls_theory=get_binned_cl(cls_full)
xis_theory=get_xi_all(cls_full)
n_cls=cls_theory.shape[0]
n_xis=xis_theory.shape[0]
cls_theory=cls_theory.flatten()
xis_theory=xis_theory.flatten()

cls_mean=np.zeros(len(cls_theory))
cls_covar=np.zeros([len(cls_theory),len(cls_theory)])
xis_mean=np.zeros(len(xis_theory))
xis_covar=np.zeros([len(xis_theory),len(xis_theory)])
for isim in np.arange(nsims) :
    print isim
    cls_sim=get_cl_sim(cls_full)

    cls_obs=get_binned_cl(cls_sim).flatten()
    xis_obs=get_xi_all(cls_sim).flatten()
    
    cls_mean+=cls_obs
    cls_covar+=cls_obs[:,None]*cls_obs[None,:]
    xis_mean+=xis_obs
    xis_covar+=xis_obs[:,None]*xis_obs[None,:]
cls_mean/=nsims
cls_covar=cls_covar/nsims-cls_mean[:,None]*cls_mean[None,:]
xis_mean/=nsims
xis_covar=xis_covar/nsims-xis_mean[:,None]*xis_mean[None,:]

plt.figure(); plt.imshow(covar_binned/np.sqrt(np.diag(covar_binned)[:,None]*np.diag(covar_binned)[None,:]),interpolation='nearest')
plt.figure(); plt.imshow(cls_covar/np.sqrt(np.diag(cls_covar)[:,None]*np.diag(cls_covar)[None,:]),interpolation='nearest')
plt.figure(); plt.imshow(xis_covar/np.sqrt(np.diag(xis_covar)[:,None]*np.diag(xis_covar)[None,:]),interpolation='nearest')
plt.show()

#C_ell file
typ,ell,t1,q1,t2,q2=[],[],[],[],[],[]
for i1 in np.arange(ntracers) :
    for i2 in np.arange(i1,ntracers) :
        ixcorr=twod_order[(i1,i2)]
        typ+=['FF']*n_ell
        ell+=list(ell_means)
        t1+=[i1]*n_ell
        t2+=[i2]*n_ell
        q1+=[tracer_types[i1]]*n_ell
        q2+=[tracer_types[i2]]*n_ell
binning_cell=sacc.Binning(typ,ell,t1,q1,t2,q2)
binning_sacc=sacc.SACC(sacc_t_list,binning_cell)
mean_cell=sacc.MeanVec(cls_theory)
prec_cell=sacc.Precision(cls_covar,mode='ell_block_diagonal', binning=binning_sacc.binning,is_covariance=True)
s=sacc.SACC(sacc_t_list,binning_cell,mean_cell,prec_cell)
s.saveToHDF(fname_cl,save_precision=True)

#Xi_th file
typ,ell,t1,q1,t2,q2=[],[],[],[],[],[]
for i1 in np.arange(ntracers) :
    sc1= (sacc_t_list[i1].type=='spin0')
    for i2 in np.arange(i1,ntracers) :
        sc2= (sacc_t_list[i2].type=='spin0')
        if sc1 and sc2 : #0-0
            for i_t,t in enumerate(th_means) :
                typ.append('+R')
                ell.append(t)
                t1.append(i1)
                t2.append(i2)
                q1.append(tracer_types[i1])
                q2.append(tracer_types[i2])
        elif sc1 or sc2 : #0-2
            for i_t,t in enumerate(th_means) :
                typ.append('+R')
                ell.append(t)
                t1.append(i1)
                t2.append(i2)
                q1.append(tracer_types[i1])
                q2.append(tracer_types[i2])
        elif (not sc1) and (not sc2) : #2-2
            for i_t,t in enumerate(th_means) :
                typ.append('+R')
                ell.append(t)
                t1.append(i1)
                t2.append(i2)
                q1.append(tracer_types[i1])
                q2.append(tracer_types[i2])
            for i_t,t in enumerate(th_means) :
                typ.append('-R')
                ell.append(t)
                t1.append(i1)
                t2.append(i2)
                q1.append(tracer_types[i1])
                q2.append(tracer_types[i2])
        else :
            print "WTF"
binning_xi=sacc.Binning(typ,ell,t1,q1,t2,q2)
binning_sacc=sacc.SACC(sacc_t_list,binning_xi)
mean_xi=sacc.MeanVec(xis_theory)
prec_xi=sacc.Precision(xis_covar,mode='dense',binning=binning_sacc.binning,is_covariance=True)
s=sacc.SACC(sacc_t_list,binning_xi,mean_xi,prec_xi)
s.saveToHDF(fname_xi,save_precision=True)
