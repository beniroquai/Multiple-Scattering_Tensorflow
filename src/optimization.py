import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy as scipy
import scipy.misc

# own functions
from src import tf_helper as tf_helper

'''
        if_optimize - Only evaluating or optimizing the system? 1/0 
        my_learningrate - e.g. 1e-4
        my_keep_prob - e.g. 1 (dropout-regularizer)
        my_tv_lambda - e.g. 1e-2 (total variatonional regularizer)
        my_positivity_constr - e.g. 10e6 (penalty for "being in the wrong regime"),
        my_sigma - e.g 0.05
        my_gr_lambda - e.g. 1e-2 (goods roughness regularizer)

'''



def loss(self, loss_type=4):
    ''' Determine the Cost-function/error-norm or fidelity term
    METHODS:
        1 -
        2 -
        3 -
        4 -

        '''

    if (self.if_optimize):
        print("Define Loss Functoind")
        # TF_total_error = tf.reduce_sum(tf_abssqr(tf.subtract(TF_allSumAmp, TF_allSumAmp_mes)))#
        # TF_total_error = tf.reduce_sum(tf.abs(tf.subtract(tf_angle(TF_allSumAmp), tf_angle(TF_allSumAmp_mes))))
        #
        self.loss_type = loss_type
        if (loss_type == 1):
            print('losstype = 1')
            # could help if more concern has to be taken on one of the two error measurements (phase/amp)
            self.TF_total_error_mag = tf.abs(
                tf.subtract(tf.abs(self.TF_allSumAmp), tf.abs(self.TF_allSumAmp_mes)))  # like Kamilov
            self.TF_total_error_ang = tf.abs(
                tf.subtract(tf_helper.tf_angle(self.TF_allSumAmp), tf_helper.tf_angle(self.TF_allSumAmp_mes)))

            self.TF_total_error_amp = tf.reduce_sum(self.TF_total_error_mag + self.TF_total_error_ang)

            # Define the error-function
            self.TF_total_error = self.TF_total_error_amp + self.TF_tv_lambda * self.TF_total_variation_loss + self.TF_obj_reg

            # optimize alternating or direct method

        elif (loss_type == 2):
            self.TF_total_error_mag = tf.reduce_sum(
                tf.abs(tf.subtract(tf.abs(self.TF_allSumAmp), tf.abs(self.TF_allSumAmp_mes))))  # like Kamilov
            self.TF_total_error_ang = tf.reduce_sum(tf.abs(
                tf.subtract(tf_helper.tf_angle(self.TF_allSumAmp), tf_helper.tf_angle(self.TF_allSumAmp_mes))))

            self.TF_total_error_mag = self.TF_total_error_mag + self.TF_reg
            self.TF_total_error_ang = self.TF_total_error_ang + self.TF_reg

            # initiliaze the TF_optimizer
            print("Define  TF_optimizer")
            # TF_optimizer = tf.train.GradientDescentTF_optimizer(learning_rate=my_learningrate).minimize(TF_total_error)
            self.TF_optimizer_mag = tf.train.AdamOptimizer(learning_rate=self.TF_learningrate).minimize(
                self.TF_total_error_mag)
            self.TF_optimizer_ang = tf.train.AdamOptimizer(learning_rate=self.TF_learningrate).minimize(
                self.TF_total_error_ang)
            # TF_optimizer = tf.train.MomentumOptimizer(learning_rate=my_learningrate, use_nesterov=True).minimize(TF_total_error)
        elif (loss_type == 3):
            #### try real and imaginary part as cost-function
            # could help if more concern has to be taken on one of the two error measurements (phase/amp)
            self.TF_total_error_mag = tf.abs(
                tf.subtract(tf.real(self.TF_allSumAmp), tf.real(self.TF_allSumAmp_mes)))  # like Kamilov
            self.TF_total_error_ang = tf.abs(
                tf.subtract(tf.imag(self.TF_allSumAmp), tf.imag(self.TF_allSumAmp_mes)))

            # optimize alternating or direct method
            if (0):
                self.TF_total_error_amp = tf.reduce_sum(self.TF_total_error_mag + self.TF_total_error_ang)

                # Define the error-function
                self.TF_total_error = self.TF_total_error_amp + self.TF_tv_lambda * self.TF_total_variation_loss + self.TF_obj_reg
            else:
                TF_total_error_mag = tf.reduce_sum(
                    tf.abs(tf.subtract(tf.abs(self.TF_allSumAmp), tf.abs(self.TF_allSumAmp_mes))))  # like Kamilov
                TF_total_error_ang = tf.reduce_sum(tf.abs(
                    tf.subtract(tf_helper.tf_angle(self.TF_allSumAmp), tf_helper.tf_angle(self.TF_allSumAmp_mes))))

                TF_total_error_mag = TF_total_error_mag + self.TF_tv_lambda * self.TF_total_variation_loss + self.TF_obj_reg
                TF_total_error_ang = TF_total_error_ang + self.TF_tv_lambda * self.TF_total_variation_loss + self.TF_obj_reg

                # initiliaze the TF_optimizer
                print("Define  TF_optimizer")
                # TF_optimizer = tf.train.GradientDescentTF_optimizer(learning_rate=TF_learningrate).minimize(TF_total_error)
                self.TF_optimizer_mag = tf.train.AdamOptimizer(learning_rate=self.TF_learningrate).minimize(
                    self.TF_total_error_mag)
                self.TF_optimizer_ang = tf.train.AdamOptimizer(learning_rate=self.TF_learningrate).minimize(
                    self.TF_total_error_ang)
                # TF_optimizer = tf.train.MomentumOptimizer(learning_rate=my_learningrate, use_nesterov=True).minimize(TF_total_error)

        elif (loss_type == 4):
            #### try real and imaginary part as cost-function
            # could help if more concern has to be taken on one of the two error measurements (phase/amp)
            self.TF_total_error_raw = tf.reduce_sum(
                (tf_helper.tf_abssqr(tf.subtract(self.TF_allSumAmp, self.TF_allSumAmp_mes))))  # like Kamilov

            # optimize alternating or direct method
            self.TF_total_error = self.TF_total_error_raw + self.TF_reg
            # self.TF_total_error = tf.cast(TF_total_error, tf.float64)
            # initiliaze the TF_optimizer
            print("Define  TF_optimizer")
            # TF_optimizer = tf.train.GradientDescentTF_optimizer(learning_rate=TF_learningrate).minimize(TF_total_error)
            # self.TF_optimizer_raw = tf.contrib.opt.ScipyOptimizerInterface(loss=TF_total_error, method='L-BFGS-B')
            self.TF_optimizer_raw = tf.train.AdamOptimizer(learning_rate=self.TF_learningrate).minimize(
                self.TF_total_error)


def optimize(self, if_evaluate, iterx, my_learningrate, my_keep_prob=1., tv_lambda=1e-5, obj_reg_lambda=1e6,
             gr_lambda=1e-5):
    if (self.if_optimize):

        if (self.loss_type == 1):
            if (not np.mod(int(iterx / 10), 2)):
                self.current_error, _ = self.sess.run([self.TF_total_error_mag, self.TF_optimizer_mag],
                                                      feed_dict={self.TF_learningrate: self.my_learningrate,
                                                                 self.TF_keepprobability: self.my_keep_prob,
                                                                 self.TF_tv_lambda: self.tv_lambda,
                                                                 self.TF_obj_reg_lambda: self.obj_reg_lambda})
                print("reduce mag " + str(self.current_error))
            else:
                self.current_error, _ = sess.run([self.TF_total_error_ang, self.TF_optimizer_ang],
                                                 feed_dict={TF_learningrate: my_learningrate,
                                                            self.TF_keepprobability: my_keep_prob,
                                                            self.TF_tv_lambda: tv_lambda,
                                                            TF_obj_reg_lambda: obj_reg_lambda})
                print("reduce ang " + str(self.current_error))
        elif (self.loss_type == 4):
            self.sess.run([self.TF_optimizer_raw],
                          feed_dict={self.TF_learningrate: my_learningrate, self.TF_keepprobability: my_keep_prob,
                                     self.TF_tv_lambda: tv_lambda, self.TF_gr_lambda: gr_lambda,
                                     self.TF_obj_reg_lambda: obj_reg_lambda})

            # TF_optimizer_raw.minimize(session=sess,  feed_dict={TF_learningrate:my_learningrate, TF_keepprobability:my_keep_prob, TF_tv_lambda:tv_lambda, TF_obj_reg_lambda:obj_reg_lambda})

        if (if_evaluate):

            self.error_total, self.np_allSumAmp, self.Obj_stack = self.sess.run(
                [self.TF_total_error, self.TF_allSumAmp, self.TF_obj],
                feed_dict={self.TF_learningrate: my_learningrate, self.TF_keepprobability: my_keep_prob,
                           self.TF_tv_lambda: tv_lambda, self.TF_gr_lambda: gr_lambda,
                           self.TF_obj_reg_lambda: obj_reg_lambda})
            print('Iteration: ' + str(iterx) + ' error_total: ' + str(self.error_total))

            # print str(TF_total_variation_loss.eval())
            # np_allSumAmp_mes = TF_allSumAmp_mes.eval()
            # plt.imshow((np.abs(np_allSumAmp_mes[:, Nx/2, :])), cmap = 'gray')
            # plt.imshow((np.abs(np_allSumAmp[:, Nx/2, :])), cmap = 'gray')

            # Visualize reconstructed object at iteration i
            np.save('Obj_stack', self.Obj_stack)

            if (1):
                scipy.misc.imsave('./results/rec-obj' + str(iterx) + '.jpg', (self.Obj_stack[:, self.Nx / 2, :]))
                scipy.misc.imsave('./results/diff_f_rec-obj' + str(iterx) + '.jpg',
                                  ((self.obj[:, self.Nx / 2, :] - self.Obj_stack[:, self.Nx / 2, :])))

                # scipy.misc.imsave('./results/mes_allsumamp_'+str(iterx)+'.jpg', (np.abs(np_allSumAmp_mes[:, Nx/2, :])))
                # scipy.misc.imsave('./results/mes_allsumamp_ang_'+str(iterx)+'.jpg', (np.angle(np_allSumAmp_mes[:, Nx/2, :])))
                scipy.misc.imsave('./results/allsumamp_' + str(iterx) + '.jpg',
                                  (np.abs(self.np_allSumAmp[:, self.Nx / 2, :])))
                scipy.misc.imsave('./results/allsumamp_ang_' + str(iterx) + '.jpg',
                                  (np.angle(self.np_allSumAmp[:, self.Nx / 2, :])))
                scipy.misc.imsave('./results/dif_allsumamp_' + str(iterx) + '.jpg', (
                            np.abs(self.allSumAmp_mes[:, self.Nx / 2, :]) - np.abs(
                        self.np_allSumAmp[:, self.Nx / 2, :])))
                scipy.misc.imsave('./results/dif_allsumang_' + str(iterx) + '.jpg', (
                            np.angle(self.allSumAmp_mes[:, self.Nx / 2, :]) - np.angle(
                        self.np_allSumAmp[:, self.Nx / 2, :])))


