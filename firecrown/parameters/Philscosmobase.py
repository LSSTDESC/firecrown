class CosmoBase(object):
    def __init__(self,
                 Omega_c, Omega_b, Omega_l, h, n_s, A_s=None, sigma_8=None,
                 Omega_g=0.0, Omega_n_mass=0.0, Omega_n_rel=0.0,
                 w0=-1., wa=0., N_nu_mass=0, N_nu_rel=3.046, mnu=0.0):
        """
        Instantiate new CosmoBase parameter class.

        Parameters
        ----------
        Omega_c, Omega_b, Omega_l, h, n_s : float
            Compulsory cosmological parameters. Density parameters are defined
            with respect to the critical density today.

        A_s, sigma_8 : float, optional
            Overall normalization of matter power spectrum. One of these must
            be specified.

        Omega_g, Omega_n_mass, Omega_n_rel : float, optional
            Optional density parameters.

        w0, wa : float, optional
            Dark energy equation of state parameters.
            Defaults: w0, wa = (-1, 0)

        N_nu_mass, N_nu_rel, mnu : float, optional
            Neutrino parameters (number of massive and relativistic neutrino
            species, and sum of neutrino masses). Defaults: (0, 3.046, 0)
        """
        # Set parameter values
        self.Omega_c = Omega_c
        self.Omega_b = Omega_b
        self.Omega_l = Omega_l
        self.h = h
        self.n_s = n_s
        self.A_s = A_s
        self.sigma_8 = sigma_8
        self.Omega_g = Omega_g
        self.Omega_n_mass = Omega_n_mass
        self.Omega_n_rel = Omega_n_rel
        self.w0 = w0
        self.wa = wa
        self.N_nu_mass = N_nu_mass
        self.N_nu_rel = N_nu_rel
        self.mnu = mnu
        self.sigma_8 = sigma_8

        # Set density parameters according to consistency relations
        self.Omega_m = self.Omega_c + self.Omega_b + self.Omega_n_mass
        self.Omega_k = 1. - (self.Omega_m + self.Omega_l
                             + self.Omega_g + self.Omega_n_rel)

    def validate(self):
        """
        Ensure that parameter values are valid and consistent. This is a basic
        physical validity check; this will not check if the parameters are
        within valid ranges for the programs that will use them.
        """
        # Positivity checks
        assert(self.Omega_c >= 0.)
        assert(self.Omega_b >= 0.)
        assert(self.Omega_l >= 0.)
        assert(self.Omega_m >= 0.)
        assert(self.Omega_n_mass >= 0.)
        assert(self.Omega_n_rel >= 0.)
        assert(self.Omega_g >= 0.)
        assert(self.h >= 0.)
        assert(self.A_s >= 0.)
        assert(self.n_s >= 0.)
        assert(self.N_nu_mass >= 0.)
        assert(self.N_nu_rel >= 0.)
        if self.sigma_8 is not None:
            assert(self.sigma_8 >= 0.)

        # Density parameters: Consistency relations
        assert(self.Omega_m == (
            self.Omega_b + self.Omega_c + self.Omega_n_mass))
        assert(self.Omega_k == 1. - (self.Omega_m + self.Omega_l + self.Omega_g
                                     + self.Omega_n_rel))

    def __str__(self):
        s = ""

        # Density parameters
        s += "Omega_m:      %4.4f\n" % self.Omega_m
        s += "Omega_c:      %4.4f\n" % self.Omega_c
        s += "Omega_b:      %4.4f\n" % self.Omega_b
        s += "Omega_l:      %4.4f\n" % self.Omega_l
        s += "Omega_k:      %4.4e\n" % self.Omega_k
        s += "Omega_n_mass: %3.3e\n" % self.Omega_n_mass
        s += "Omega_n_rel:  %3.3e\n" % self.Omega_n_rel
        s += "Omega_g:      %3.3e\n" % self.Omega_g

        # Hubble
        s += "h:            %3.3e\n" % self.h

        # Dark energy equation of state parameters
        s += "w0:           %3.3f\n" % self.w0
        s += "wa:           %3.3f\n" % self.wa

        # Neutrino properties
        s += "N_nu_mass:    %3.3f\n" % self.N_nu_mass
        s += "N_nu_rel:     %3.3f\n" % self.N_nu_rel
        s += "mnu:          %4.4f\n" % self.mnu

        # Primordial power spectrum parameters
        s += "A_s:          %4.4e\n" % self.A_s
        s += "n_s:          %4.4f\n" % self.n_s
        if self.sigma_8 is not None:
            s += "sigma_8:      %4.4f\n" % self.sigma_8
        else:
            s += "sigma_8:      nan\n"
        return s
