#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ps-6
"""

from astropy.io import fits
# documentation: see https://docs.astropy.org/en/stable/io/fits/
import matplotlib.pyplot as plt
import numpy as np
import time
hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

#a
plt.plot(logwave, flux[0, :],label='galaxy1')
plt.plot(logwave, flux[2, :],label='galaxy2')
plt.plot(logwave, flux[3, :],label='galaxy3')
plt.plot(logwave, flux[4, :],label='galaxy4')
plt.ylabel('flux [$10^{−17}$ erg s$^{−1}$ cm$^{−2}$ A$^{-1}$]', fontsize = 16)
plt.xlabel('logwavelength [$A$]', fontsize = 16)
plt.legend()
plt.savefig('a.png')
plt.show()

#b normalization
flux_sum = np.sum(flux, axis = 1)
flux_normalized = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T

plt.plot(np.sum(flux_normalized, axis = 1))
plt.ylim(0,2)
plt.ylabel('normalization', fontsize = 16)
plt.xlabel('galaxy', fontsize = 16)
plt.savefig('b.png')
plt.show()

#c
means_normalized = np.mean(flux_normalized, axis=1)
flux_normalized_0_mean = flux_normalized-np.tile(means_normalized, (np.shape(flux)[1], 1)).T
plt.plot(logwave, flux_normalized_0_mean[0, :],label='galaxy1')
plt.plot(logwave, flux_normalized_0_mean[2, :],label='galaxy2')
plt.plot(logwave, flux_normalized_0_mean[3, :],label='galaxy3')
plt.plot(logwave, flux_normalized_0_mean[4, :],label='galaxy4')
plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('logwavelength [$A$]', fontsize = 16)
plt.legend()
plt.savefig('c.png')
plt.show()

#d
def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec

r = flux_normalized_0_mean
C = r.T@r
print(C.shape)

start_np = time.time()
eigvals, eigvecs = sorted_eigs(r, return_eigvalues = True)
end_np = time.time()
[plt.plot(eigvecs[i],label=i)for i in range(5)]
plt.legend()
plt.ylabel('eigenvector', fontsize = 16)
plt.xlabel('wavelength basis', fontsize = 16)
plt.savefig('d.png')
plt.show()

#e
start_svd = time.time()
U, S, Vh = np.linalg.svd(r, full_matrices=True)
# rows of Vh are eigenvectors
eigvecs_svd = Vh.T
eigvals_svd = S**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]
end_svd = time.time()
[plt.plot(eigvecs_svd[i],label=i)for i in range(5)]
plt.legend()
plt.ylabel('eigenvector', fontsize = 16)
plt.xlabel('wavelength basis', fontsize = 16)
plt.savefig('e.png')
plt.show()

print("Running time for np method is", end_np - start_np)
print("unning time for svd method is", end_svd - start_svd)

#f
print("condition number for C is", np.max(S**2)/np.min(S**2))
print("condition number for R is", np.max(S)/np.min(S))

#g
def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
appro_spectra = PCA(5,r,project = True)
plt.plot(logwave, appro_spectra[0,:], label = 'l = 5')
plt.plot(logwave, r[0,:], label = 'original data')
plt.ylabel('normalized 0-mean flux', fontsize = 16)
plt.xlabel('logwavelength [$A$]', fontsize = 16)
plt.legend()
plt.savefig('g.png')
plt.show()

#h
w = PCA(5,r,project = False)
c_0 = w[:,0].T
c_1 = w[:,1].T
c_2 = w[:,2].T
plt.plot(c_0, c_1, '.',label = 'c_0 vs c_1')
plt.plot(c_0, c_2, '.',label = 'c_0 vs c_2')
plt.ylabel('c_1 or c_2', fontsize = 16)
plt.xlabel('c_0', fontsize = 16)
plt.legend()
plt.savefig('h.png')
plt.show()

#i
sfr = np.zeros(20)
for l in range(1,21):
    appro_spectra = PCA(l,r,project = True)
    sfr[l-1] = np.sum((appro_spectra[0,:]-r[0,:])**2)
plt.plot(range(1,21),sfr)
plt.ylabel('squared fractional residuals', fontsize = 16)
plt.xlabel('reduced dimension $N_c$', fontsize = 16)
plt.savefig('i.png')
plt.show()

print("the fractional error for Nc = 20 is", sfr[19])




