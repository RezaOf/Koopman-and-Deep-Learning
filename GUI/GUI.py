import sys
import numpy as np
import matplotlib.pylab as plt
import pickle
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PendulumODEsolver import getPendulumTraj


import notebookfns as nf


class TrainedNN:
    def __init__(self, fname):
        with open(fname, 'rb') as f:
            self.params = pickle.load(f, encoding='latin1')
        self.W, self.b = nf.load_weights_koopman(fname, len(self.params['widths'])-1,
                                                 len(self.params['widths_omega_real'])-1,
                                                 self.params['num_real'],
                                                 self.params['num_complex_pairs'])

class SampleTraj:
    def __init__(self, Tspan, Xode, X):
        self.Tspan = Tspan
        self.Xode = Xode
        self.X = X

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("GUI_Qt.ui", self)

        self.Trajfig, self.Trajax = plt.subplots(3,1)
        self.TrajPlotWidget = FigureCanvas(self.Trajfig)
        Trajlayout = QtWidgets.QVBoxLayout(self.MPcanvas4traj)
        Trajlayout.setContentsMargins(0, 0, 0, 0)
        Trajlayout.addWidget(self.TrajPlotWidget)

        self.addToolBar(NavigationToolbar(self.TrajPlotWidget, self))

        self.PPfig, self.PPax = plt.subplots()
        self.PPplotWidget = FigureCanvas(self.PPfig)
        PPlayout = QtWidgets.QVBoxLayout(self.MPcanvas4PP)
        PPlayout.setContentsMargins(0, 0, 0, 0)
        PPlayout.addWidget(self.PPplotWidget)
        #  self.addToolBar(NavigationToolbar(self.PPplotWidget, self))

        self.x0Validator = QDoubleValidator(-90, 90, 3)
        self.x0Validator.setNotation(0)
        self.x0in.setValidator(self.x0Validator)

        self.x1Validator = QDoubleValidator(-90, 90, 3)
        self.x1Validator.setNotation(0)
        self.x1in.setValidator(self.x1Validator)

        self.EvalTraj.clicked.connect(self.SolveTraj)

        fname = '../Training/TrainedNetwork/Original/Pendulum_2019_03_30_10_28_04_700272_model.pkl'
        self.OrigNN = TrainedNN(fname)

        fname = '../Training/TrainedNetwork/Improved/Pendulum_2019_04_23_00_29_16_067139_model.pkl'
        self.ImprovedNN = TrainedNN(fname)



    def SolveTraj(self):
        IC = np.array([float(self.x0in.text()), float(self.x1in.text())]) * np.pi/180
        tEnd = 13

        # Improved Network
        num_steps = np.int(tEnd / self.ImprovedNN.params['delta_t'])
        Tspan = np.linspace(0, tEnd, num_steps+1)
        X = np.zeros((num_steps+1, 2))
        X[0, :] = IC
        X[1:, :] = nf.PredictKoopmanNetOmegas(IC, self.ImprovedNN.W, self.ImprovedNN.b,
                                              self.ImprovedNN.params['delta_t'], num_steps,
                                              self.ImprovedNN.params['num_real'],
                                              self.ImprovedNN.params['num_complex_pairs'],
                                              self.ImprovedNN.params['num_encoder_weights'],
                                              self.ImprovedNN.params['num_omega_weights'],
                                              self.ImprovedNN.params['num_decoder_weights'])
        X = X * 180/np.pi
        Xode, temp = getPendulumTraj(IC, tEnd, num_steps)
        Xode = Xode * 180 / np.pi
        ImprovedTraj = SampleTraj(Tspan, Xode, X)

        # Original Network
        num_steps = np.int(tEnd / self.OrigNN.params['delta_t'])
        Tspan = np.linspace(0, tEnd, num_steps+1)
        X = np.zeros((num_steps+1, 2))
        X[0, :] = IC
        X[1:, :] = nf.PredictKoopmanNetOmegas(IC, self.OrigNN.W, self.OrigNN.b,
                                              self.OrigNN.params['delta_t'], num_steps,
                                              self.OrigNN.params['num_real'],
                                              self.OrigNN.params['num_complex_pairs'],
                                              self.OrigNN.params['num_encoder_weights'],
                                              self.OrigNN.params['num_omega_weights'],
                                              self.OrigNN.params['num_decoder_weights'])
        X = X * 180/np.pi
        Xode, temp = getPendulumTraj(IC, tEnd, num_steps)
        Xode = Xode * 180 / np.pi
        OrigTraj = SampleTraj(Tspan, Xode, X)

        # Now Plot
        self.plot(ImprovedTraj, OrigTraj)
        return 1

    def plot(self, ImprovedTraj, OrigTraj):
            self.Trajax[0].cla()
            self.Trajax[0].plot(ImprovedTraj.Tspan, ImprovedTraj.X[:, 0], 'b-', label=r'$Improved\, NN:\, \theta\,(deg)$', linewidth=3.0)
            self.Trajax[0].plot(ImprovedTraj.Tspan, ImprovedTraj.Xode[::1, 0], 'r--', label=r'ode45: $\theta\,(deg)$')  #\dot{\theta}\,(\frac{deg}{sec})
            self.Trajax[0].plot(ImprovedTraj.Tspan, ImprovedTraj.X[:, 1], 'k-', label=r'$Improved\, NN:\, \dot{\theta}\,(\frac{deg}{sec})$',linewidth=3.0)
            self.Trajax[0].plot(ImprovedTraj.Tspan, ImprovedTraj.Xode[::1, 1], 'c--', label=r'ode45: $\dot{\theta}\,(\frac{deg}{sec})$')

            self.Trajax[0].set_ylabel('$Response$')
            self.Trajax[0].set_xlabel('$Time\,(sec)$')
            self.Trajax[0].grid()
            self.Trajax[0].legend(loc=4, framealpha=0.4)

            self.Trajax[1].cla()
            self.Trajax[1].plot(ImprovedTraj.Tspan, ImprovedTraj.X[:, 0] - ImprovedTraj.Xode[::1, 0], 'r-', label=r'Improved $\theta\,(deg)$', linewidth=2.0)
            self.Trajax[1].plot(OrigTraj.Tspan, OrigTraj.X[:, 0] - OrigTraj.Xode[::1, 0], 'b-', label=r'Orig $\theta\,(deg)$', linewidth=2.0)

            self.Trajax[1].set_ylabel('$Errors\,\, (Trained NN-ode45)$')
            self.Trajax[1].set_xlabel('$Time\,(sec)$')
            self.Trajax[1].grid()
            self.Trajax[1].legend(loc='best', framealpha=0.4)

            self.Trajax[2].cla()
            self.Trajax[2].plot(ImprovedTraj.Tspan, ImprovedTraj.X[:, 1] - ImprovedTraj.Xode[::1, 1], 'r-', label=r'Improved $\dot{\theta}\,(\frac{deg}{sec})$', linewidth=2.0)
            self.Trajax[2].plot(OrigTraj.Tspan, OrigTraj.X[:, 1] - OrigTraj.Xode[::1, 1], 'b-', label=r'Orig $\dot{\theta}\,(\frac{deg}{sec})$', linewidth=2.0)

            self.Trajax[2].set_ylabel('$Errors\,\, (Trained NN-ode45)$')
            self.Trajax[2].set_xlabel('$Time\,(sec)$')
            self.Trajax[2].grid()
            self.Trajax[2].legend(loc='best', framealpha=0.4)
            self.Trajfig.tight_layout()

            self.PPax.cla()
            self.PPax.plot(ImprovedTraj.X[:, 0], ImprovedTraj.X[:, 1], 'b-', label=r'$Improved\, NN$', linewidth=3.0)
            self.PPax.plot(ImprovedTraj.Xode[:, 0], ImprovedTraj.Xode[:, 1], 'r--', label=r'$Ode45$')
            self.PPax.set_ylabel(r'$\dot{\theta}$')
            self.PPax.set_xlabel(r'$\theta$')
            self.PPax.text(0.45, 0.55, r'$Phase$ $Plane$', transform=self.PPfig.transFigure)
            self.PPax.axis('equal')
            self.PPax.grid()
            self.PPax.legend(loc='best', framealpha=0.4)
            self.PPfig.tight_layout()

            self.Trajfig.canvas.draw()
            self.PPfig.canvas.draw()
            return 0



app = QtWidgets.QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.show()
sys.exit(app.exec_())
