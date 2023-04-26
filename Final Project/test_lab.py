import random
import time 
import unittest

import numpy as np
from scipy import signal

import linear_algebra

def is_upper_triangular(R,tol=1e-10):
    for i in range(1, len(R)):
        for j in range(0, i):
            if(abs(R[i][j]) > tol):
                    return False
    return True


class TestLinalg(unittest.TestCase):
    def test_cheating(self):
        print("\nTesting if np.linalg was used")

        with open("Final Project\linear_algebra.py", "r") as file:
            content = file.read()

        eigen_check = False
        # check if string present or not
        if "eig(" in content:
            eigen_check = True

        self.assertFalse(eigen_check,
                         "You are illegally using "
                         "the built-in eig function")

        qr_check = False
        if "qr(" in content:
            qr_check = True

        self.assertFalse(qr_check,
                         "You are illegally using "
                         "the built-in qr function")

        linalg_check = False
        if "linalg" in content:
            linalg_check = True

        self.assertFalse(linalg_check,
                         "You are illegally using "
                         "the np.linalg module")

        
    def test_qr(self):
        print("\nTesting qr fidelity")

        for i in range(20):
            N = np.random.randint(3,7)
            M = np.random.randint(N,8)

            A = np.random.uniform(-100,100,size=(M,N))
            
            Q, R = linear_algebra.qrgs(A)
            
            A_qr = Q@R
            np.testing.assert_allclose(
                A_qr, A,
                err_msg="A = Q@R test failed"
            )

            I = Q.T @ Q
            np.testing.assert_array_almost_equal(
                I, np.eye(N),
                err_msg="I = Q.T@Q test failed"
            )
        
            self.assertTrue(is_upper_triangular(R),
                            "R is not upper triangular")

    def test_leastsquares(self):
        print("\nTesting least squares")

        for _ in range(20):
            M = np.random.randint(100,150)
            N = np.random.randint(5,20)
        
            A = np.random.uniform(-100,100,size=(M,N))
            b = np.random.uniform(-100,100,size=(M))
            
            x_my = linear_algebra.linear_least_squares(A,b)
            x_np, _, _, _ = np.linalg.lstsq(A,b,rcond=-1)
            
            np.testing.assert_allclose(
                x_my, x_np,
                err_msg="Ax=b solver test failed")
            
            
    def test_eigenvalues(self):
        print("\nTesting eigenvalue fidelity")

        for i in range(20):
            N = np.random.randint(3,6)
            A = np.random.uniform(-100,100,size=(N,N))
            A = (A.T + A)/2
            w_my, _ = linear_algebra.eigen(A)
            w_sol, _ = np.linalg.eig(A)
            np.testing.assert_allclose(sorted(w_my),
                                       sorted(w_sol))

    #increasing tolerance so we can test eigen vectors, uncomment above and comment below to go back to normal
    # def test_eigenvalues(self):
    #     print("\nTesting eigenvalue fidelity")

    #     for i in range(20):
    #         N = np.random.randint(3,6)
    #         A = np.random.uniform(-100,100,size=(N,N))
    #         A = (A.T + A)/2
    #         w_my, _ = linear_algebra.eigen(A)
    #         w_sol, _ = np.linalg.eig(A)
    #         np.testing.assert_allclose(sorted(w_my),
    #                                    sorted(w_sol),
    #                                    rtol = 1)

    def test_eigenvectors(self):
        print("\nTesting eigenvector fidelity")
        
        for i in range(20):
            N = np.random.randint(3,6)
            A = np.random.uniform(-100,100,size=(N,N))
            A = (A.T + A)/2
            w_my, v_my = linear_algebra.eigen(A)
            w_np, v_np = np.linalg.eig(A)
            
            v_my = v_my[:,np.argsort(w_my)]
            v_np = v_np[:,np.argsort(w_np)]

            for i in range(v_my.shape[0]):
                if v_my[0,i] < 0:
                    v_my[:,i] *= -1
                if v_np[0,i] < 0:
                    v_np[:,i] *= -1
                    
            np.testing.assert_allclose(v_my,
                                       v_np)

    #increasing tolerance so we can test eigen vectors, uncomment above and comment below to go back to normal
    # def test_eigenvectors(self):
    #     print("\nTesting eigenvector fidelity")
        
    #     for i in range(20):
    #         N = np.random.randint(3,6)
    #         A = np.random.uniform(-100,100,size=(N,N))
    #         A = (A.T + A)/2
    #         w_my, v_my = linear_algebra.eigen(A)
    #         w_np, v_np = np.linalg.eig(A)
            
    #         v_my = v_my[:,np.argsort(w_my)]
    #         v_np = v_np[:,np.argsort(w_np)]

    #         for i in range(v_my.shape[0]):
    #             if v_my[0,i] < 0:
    #                 v_my[:,i] *= -1
    #             if v_np[0,i] < 0:
    #                 v_np[:,i] *= -1
                    
    #         np.testing.assert_allclose(v_my,
    #                                    v_np,
    #                                    rtol = 1)
      
         
if __name__ == "__main__":
    unittest.main()
