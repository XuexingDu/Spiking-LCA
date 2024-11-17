import brainpy as bp
import brainpy.math as bm
import numpy as np


class BaseExpSyn(bp.SynConn):
    def __init__(self, pre, post, conn, g_max=1.0, tau=1.0, method='rk2'):
        super(BaseExpSyn, self).__init__(pre=pre, post=post, conn=conn)

        # check whether the pre group has the needed attribute: "spike"
        self.check_pre_attrs('spike')

        # check whether the post group has the needed attributes: "input" and "V"
        self.check_post_attrs('input', 'V')

        # parameters
        self.tau = tau
        self.g_max = g_max
        self.spike_matrix = self.pre.cross

        # integral function
        self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

class ExpConnMat(BaseExpSyn):
    def __init__(self, *args, **kwargs):
        super(ExpConnMat, self).__init__(*args, **kwargs)

        # connection matrix
        self.conn_mat = self.conn.require('conn_mat').astype(float)

        # synapse gating variable
        # -------
        # NOTE: Here the synapse number is the same with
        #       the post-synaptic neuron number. This is
        #       different from the AMPA synapse.
        self.g = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi):
        _t, _dt = tdi.t, tdi.dt
        # integrate the synapse state
        self.g.value = self.integral(self.g, _t, dt=_dt)
        # update synapse states according to the pre spikes
        # get the post-synaptic current
        post_sps = bm.dot(self.spike_matrix.astype(float), self.g_max)
        self.g += post_sps
        self.post.input += self.g

class IF_rk2(bp.NeuGroupNS):
    def __init__(self, size, b, lamb,  **kwargs):
        super(IF_rk2, self).__init__(size, **kwargs)

        self.V = bm.Variable(bm.zeros(size))
        self.cross = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))
        self.b     = b
        self.lamb  = lamb

    def update(self):
        dt = bp.share.load('dt')
        V = self.V + dt * (self.b + self.input - self.lamb)
        self.spike.value = V >= 1.
        dt_cross = bm.where(self.spike > 0., dt * (1 - self.V)/(V - self.V), 0.)
        self.cross.value = bm.where(self.spike >0., bm.exp(dt_cross -dt), 0.)
        # self.cross.value = dt_cross
        self.V.value = bm.where(self.spike > 0., 0. - (dt_cross - dt) * (self.b + self.input -self.lamb)  , V)
        self.input[:] = 0


class SLCA_rk2(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, scale=1.0, method='rk2'):
        super(SLCA_rk2, self).__init__()
        
        # network size
        num_neuron = int(num * scale)
        self.N   = IF_rk2(num_neuron,b,lamb)

        # synapses
        self.w_matrix = w   # excitatory synaptic weight (voltage)
        # w_matrix  = 0.
        self.N2N = ExpConnMat(pre=self.N,
                                post=self.N,
                                conn= bp.connect.All2All(),
                                tau=1.,
                                g_max= -self.w_matrix,
                                method=method)
        
    def update(self):
        t = bp.share.load('t')
        dt = bp.share.load('dt')
        self.N2N()
        self.N()


class SLCA_rk2_double(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, scale=1.0, method='rk2'):
        super(SLCA_rk2_double, self).__init__()
        
        # network size
        num_neuron = int(num * scale)
        self.N_ex   = IF_rk2(num_neuron,b,lamb)
        self.N_in   = IF_rk2(num_neuron,-b,lamb)


        # synapses
        self.w_matrix = w   # excitatory synaptic weight (voltage)
        # w_matrix  = 0.
        self.N_ex2N_ex = ExpConnMat(pre=self.N_ex,
                                post=self.N_ex,
                                conn= bp.connect.All2All(),
                                tau=1.,
                                g_max= -self.w_matrix,
                                method=method)
        
        self.N_ex2N_in = ExpConnMat(pre=self.N_ex,
                                    post=self.N_in,
                                    conn= bp.connect.All2All(),
                                    tau=1.,
                                    g_max= self.w_matrix + bm.eye(num_neuron),
                                    method=method)     
        
        self.N_in2N_ex = ExpConnMat(post=self.N_ex,
                            pre=self.N_in,
                            conn= bp.connect.All2All(),
                            tau=1.,
                            g_max= self.w_matrix + bm.eye(num_neuron),
                            method=method) 
        
        self.N_in2N_in = ExpConnMat(post=self.N_in,
                            pre=self.N_in,
                            conn= bp.connect.All2All(),
                            tau=1.,
                            g_max= -self.w_matrix,
                            method=method) 

        
    def update(self):
        self.N_ex2N_ex()
        self.N_ex2N_in()
        self.N_in2N_ex()
        self.N_in2N_in()
        self.N_ex()
        self.N_in()


# IF model
class IF(bp.NeuGroupNS):
    def __init__(self, size, b, lamb, **kwargs):
        super(IF, self).__init__(size, **kwargs)
        self.V = bm.Variable(bm.zeros(size))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))
        self.b = b
        self.lamb  = lamb

    def update(self):
        dt = bp.share.load('dt')
        t  = bp.share.load('t')
        
        V = self.V + dt * (self.b + self.input -self.lamb)
        self.spike.value = V >= 1.
        self.V.value = bm.where(self.spike > 0., 0.  , V)
        self.input[:] = 0


class SLCA_IF(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, method='rk2', **kwargs):
        super(SLCA_IF, self).__init__( **kwargs)

        num_neuron = int(num)
        self.N = IF(num_neuron, b ,lamb)
        self.firing_rate = bm.Variable(bm.zeros(num_neuron))
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= -w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method)

    def update(self):
        self.N2N()
        self.N()

        
# LIF model
class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, V_rest=0., V_reset=0., V_th=1., R=100., tau=100., t_ref= 0.05, name=None):
        # 初始化父类
        super(LIF, self).__init__(size=size, name=name)

        # 初始化参数
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_ref = t_ref  # 不应期时长

        # 初始化变量
        self.V = bm.Variable(bm.zeros(self.num) + V_reset)
        self.input = bm.Variable(bm.zeros(self.num))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)  # 上一次脉冲发放时间
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))  # 是否处于不应期
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态
        # self.Iext  = bm.Variable(bm.zeros(self.num))

        # 使用指数欧拉方法进行积分
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, t, R, Iext):
        dvdt = (-V + self.V_rest + R * Iext) / self.tau
        return dvdt
    
    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        # 以数组的方式对神经元进行更新
        refractory = (t - self.t_last_spike) <= self.t_ref  # 判断神经元是否处于不应期
        V = self.integral(self.V, t, self.R, self.input, dt=dt)  # 根据时间步长更新膜电位
        V = bm.where(refractory, self.V, V)  # 若处于不应期，则返回原始膜电位self.V，否则返回更新后的膜电位V
        spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
        self.spike.value = spike  # 更新神经元脉冲发放状态
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)  # 更新最后一次脉冲发放时间
        self.V.value = bm.where(spike, self.V_reset, V)  # 将发放了脉冲的神经元膜电位置为V_reset，其余不变
        self.refractory.value = bm.logical_or(refractory, spike)  # 更新神经元是否处于不应期
        self.input[:] = 0.  # 重置外界输入

class SLCA_LIF(bp.Network):
    def __init__(self, num, w, b, lamb, V_rest=0., V_reset=0., V_th=1., R=10., tau=10., t_ref= 0.05, scale=1.0, method='exp_auto'):
        super(SLCA_LIF, self).__init__()
        
        # parameter setting
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_ref = t_ref  # 不应期时长
        self.lamb  = lamb
        self.b     = b
        
        # network size
        num_neuron = int(num * scale)

        pars = dict(V_rest=V_rest, V_th=V_th, V_reset=V_reset, tau = tau, R =R, t_ref = t_ref)
        self.N = LIF(num_neuron, **pars)
        self.average_current = bm.Variable(bm.zeros(num_neuron))          # 平均电流记录
        # synapses
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method,
                                            comp_method='dense')
        
    def update(self,tdi):
        t, dt = tdi.t, tdi.dt
        self.N2N(tdi)
        self.average_current.value = self.average_current *t/(t + 1) + (self.b - self.N2N.g)/(t+1)
        firing_rate =  bm.maximum(self.average_current - self.lamb, 0.)
        # self.N.input = bm.where(firing_rate > 1e-7, self.V_th / (self.R * (1 - bm.exp((self.t_ref - 1 / firing_rate) / self.R))), self.V_th/self.R)
        self.N.input   = bm.where(firing_rate > 1e-7, self.V_th / (self.R * (1 - bm.exp((self.t_ref - 1 / firing_rate) / self.R))), self.V_th/self.R)
        self.N(tdi)

# Izh model
class Izh(bp.NeuGroupNS):
    def __init__(self, size, a=0.02, b=-0.1, c=-55.0, d= 6.0, V_th=30., method='rk2', **kwargs):
        super(Izh, self).__init__(size=size, **kwargs)
        # parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V_th = V_th
        
        # 初始化变量
        self.V = bm.Variable(bm.zeros(self.num) - 65.)  # 膜电位
        self.u = bm.Variable(self.V * b)  # u变量
        self.input = bm.Variable(bm.zeros(self.num))  # 外界输入
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))  # 脉冲发放状态
        
        # 定义积分器
        self.integral = bp.odeint(f=self.derivative, method=method)
            
    def dV(self, V, t, u, Iext):
        return 0.04 * V * V + 5 * V + 140 - u + Iext * 100
    
    def du(self, u, t, V):
        return self.a * (self.b * V - u)
    
    # 将两个微分方程联合为一个，以便同时积分
    @property
    def derivative(self):
        return bp.JointEq([self.dV, self.du])

    def update(self):
        _t = bp.share.load('t')
        _dt = bp.share.load('dt')
        V, u = self.integral(self.V, self.u, _t, self.input, dt=_dt)  # 更新变量V, u
        spike = V > self.V_th  # 将大于阈值的神经元标记为发放了脉冲
        self.spike.value = spike  # 更新神经元脉冲发放状态
        self.V.value = bm.where(spike, self.c, V)  # 将发放了脉冲的神经元的V置为c，其余不变
        self.u.value = bm.where(spike, u + self.d, u)  # 将发放了脉冲的神经元的u增加d，其余不变
        self.input[:] = 0.  # 重置外界输入

class SLCA_Izh(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, method='exp_auto'):
        super(SLCA_Izh, self).__init__()
        
        # parameter setting
        self.lamb = lamb
        self.b    = b
        
        # network size
        num_neuron = int(num)
        self.N = Izh(num_neuron)
        self.average_current = bm.Variable(bm.zeros(num_neuron))          # 平均电流记录
        # synapses
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method)

    def reversed_function(self,x):
        return 0.24999688 *x**2 + 2.86516435*x + 0.20432898
                
    def update(self):
        t = bp.share.load('t')
        self.N2N()
        self.average_current.value = self.average_current *t/(t + 1) + (self.b - self.N2N.g)/(t+1)
        firing_rate  =  bm.maximum(self.average_current - self.lamb, 0.)
        self.N.input =  bm.where(firing_rate > 1e-7, self.reversed_function(firing_rate) , 0)
        self.N()


# ML model
class ML_Scale(bp.NeuGroup):
  def __init__(self, size, method='rk2', **kwargs):
    super(ML_Scale, self).__init__(size=size, **kwargs)
    # parameters
    self.V_Ca = 120./5.
    self.g_Ca = 4.
    self.V_K = -84./5.
    self.g_K = 8.
    self.V_leak = -60./5.
    self.g_leak = 2.
    self.C = 2./5.
    self.V1 = -1.2
    self.V2 = 18
    self.V_th = -15./5.


    self.V3 = 12.
    self.V4 = 17.
    self.phi = 2/3

    # variables
    self.V = bm.Variable(-40*bm.ones(self.num))
    self.W = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # functions
    def dV(V, t, W, input):
        M_inf = (1 / 2) * (1 + bm.tanh((V*5. - self.V1) / self.V2))
        I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
        I_K = self.g_K * W * (V - self.V_K)
        I_Leak = self.g_leak * (V - self.V_leak)
        dVdt = (- I_Ca - I_K - I_Leak + input*2.) / self.C
        return dVdt

    def dW(W, t, V):
        tau_W = 1 / (self.phi * 5. * bm.cosh((V*5. - self.V3) / (2 * self.V4)))
        W_inf = (1 / 2) * (1 + bm.tanh((V*5. - self.V3) / self.V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    self.int_V = bp.odeint(dV, method=method)
    self.int_W = bp.odeint(dW, method=method)

  def update(self, tdi):
    V = self.int_V(self.V, tdi.t, self.W, self.input, tdi.dt)
    self.W.value = self.int_W(self.W, tdi.t, self.V, tdi.dt)
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.input[:] = 0.

class SLCA_ML(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, method='exp_auto'):
        super(SLCA_ML, self).__init__()
        
        # parameter setting
        self.lamb = lamb
        self.b   = b
        
        # network size
        num_neuron = int(num)
        self.N = ML_Scale(num_neuron)
        self.average_current = bm.Variable(bm.zeros(num_neuron))          # 平均电流记录
        # synapses
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method)

    # def reversed_function(self,x):
    #     return 7.43164314*x**4 -14.61153623* x**3 + 13.71690213 *x**2 -3.86053151*x + 4.32149502

    def reversed_function(self, x):
        return 8.35304086 * x**5 - 22.99371634 * x**4 + 26.47121174 * x**3 - 11.32892117 * x**2 + 2.74156822 * x + 3.76677658
               
    def update(self):
        t = bp.share.load('t')
        self.N2N()
        self.average_current.value = self.average_current *t/(t + 1) + (self.b - self.N2N.g)/(t+1)
        firing_rate  =  bm.maximum(self.average_current - self.lamb, 0.)
        self.N.input =  bm.where(firing_rate > 1e-7, self.reversed_function(firing_rate) , 3.9)
        self.N()

# WB model
class WB_Scale(bp.NeuGroupNS):
  def __init__(self, size, ENa=5.5, gNa=35., EK=-9.0, gK= 9., EL=-6.5, gL=0.1,
               V_th=2.0,phi = 5.0, C= 0.1, method='rk2', **kwargs):
    # providing the group "size" information
    super(WB_Scale, self).__init__(size=size, **kwargs)

    # initialize parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.phi = phi
    self.V_th = V_th

    # initialize variables
    self.V = bm.Variable(-6.5*bm.ones(self.num))
    self.h = bm.Variable(0.6 * bm.ones(self.num))
    self.n = bm.Variable(0.32 * bm.ones(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integral functions
    self.int_V = bp.odeint(f=self.dV, method=method)
    self.int_h = bp.odeint(f=self.dh, method=method)
    self.int_n = bp.odeint(f=self.dn, method=method)

  def m_inf(self, V):
    alpha = -0.01 * ((V + 3.5)*10) / (bm.exp(-0.1 * ((V + 3.5)*10)) - 1)
    beta = 0.4 * bm.exp(-((V + 6.)*10.) / 18.)
    return alpha / (alpha + beta)

  def dh(self, h, t, V):
    alpha = 0.7 * bm.exp(-((V + 5.8)*10) / 20)
    beta = 10. / (bm.exp(-0.1 * ((V + 2.8)*10)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return self.phi * dhdt

  def dn(self, n, t, V):
    alpha = -0.1 * ((V + 3.4)*10) / (bm.exp(-0.1 * ((V + 3.4)*10)) - 1)
    beta = 1.25 * bm.exp(-((V + 4.4)*10) / 80)
    dndt = alpha * (1 - n) - beta * n
    return self.phi * dndt

  def dV(self, V, t, h, n, I_ext):
    INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
    IK = self.gK * n ** 4 * (V - self.EK)
    IL = self.gL * (V - self.EL)
    dVdt = (- INa - IK - IL + I_ext) / self.C
    return dVdt
  

  def update(self):
    t = bp.share.load('t')
    dt = bp.share.load('dt')
    # compute V, m, h, n
    V = self.int_V(self.V, t, self.h, self.n, self.input, dt=dt)
    self.h.value = self.int_h(self.h, t, self.V, dt=dt)
    self.n.value = self.int_n(self.n, t, self.V, dt=dt)

    # update the spiking state and the last spiking time
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    # update V
    self.V.value = V

    # reset the external input
    self.input[:] = 0.

class SLCA_WB(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, method='exp_auto'):
        super(SLCA_WB, self).__init__()
        
        # parameter setting
        self.lamb = lamb
        self.b    = b
        
        # network size
        num_neuron = int(num)
        self.N = WB_Scale(num_neuron)
        self.average_current = bm.Variable(bm.zeros(num_neuron))          # 平均电流记录
        # synapses
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method)

    def reversed_function(self,x):
        return 0.00302143*x**4 -0.00390291* x**3 + 0.09295591 *x**2 + 0.18930779*x - 0.00373916
                
    def update(self):
        t = bp.share.load('t')
        self.N2N()
        self.average_current.value = self.average_current *t/(t + 1) + (self.b - self.N2N.g)/(t+1)
        firing_rate  =  bm.maximum(self.average_current - self.lamb, 0.)
        self.N.input =  bm.where(firing_rate > 1e-7, self.reversed_function(firing_rate) , 0)
        self.N()


# GIF model 
class SLCA_GIF(bp.DynamicalSystemNS):
    def __init__(self, num, w, b, lamb, method='rk2'):
        super(SLCA_GIF, self).__init__()
        
        self.lamb = lamb
        self.b    = b
        
        # network size
        num_neuron = int(num)
        self.N = bp.neurons.GIF(num_neuron, method=method)
        self.average_current = bm.Variable(bm.zeros(num_neuron))          # 平均电流记录
        # synapses
        self.N2N = bp.synapses.Exponential(pre=self.N,
                                            post=self.N,
                                            conn= bp.connect.All2All(),
                                            g_max= w,
                                            tau=1.,
                                            output=bp.synouts.CUBA(),
                                            method=method)

    def reversed_function(self, x):
        return -1.87392121 * x**5 + 6.91853151 * x**4 - 9.31760692 * x**3 + 5.73923156 * x**2 + 18.46991473 * x + 0.66899093
                
    def update(self):
        t = bp.share.load('t')
        self.N2N()
        self.average_current.value = self.average_current *t/(t + 1) + (self.b - self.N2N.g)/(t+1)
        firing_rate  =  bm.maximum(self.average_current - self.lamb, 0.)
        self.N.input =  bm.where(firing_rate > 1e-7, self.reversed_function(firing_rate) , 1.)
        self.N()
    