class UnitConversion:
    @staticmethod
    def psi_to_pa(psi: float) -> float:
        return psi * 6894.76

    @staticmethod
    def pa_to_psi(psi: float) -> float:
        return psi / 6894.76

    @staticmethod
    def mps2_to_g(mps2: float) -> float:
        return mps2 / 9.80665

    @staticmethod
    def g_to_mps2(g: float) -> float:
        return g * 9.80665

    @staticmethod
    def lbpin_to_npm(lbpin: float) -> float:
        return lbpin * 175.126835

    @staticmethod
    def npm_to_lbpin(npm: float) -> float:
        return npm / 175.126835

    @staticmethod
    def in_to_m(inches: float) -> float:
        return inches * 0.0254

    @staticmethod
    def m_to_in(m: float) -> float:
        return m / 0.0254

    @staticmethod
    def lb_to_kg(lb: float) -> float:
        return lb * 0.453592

    @staticmethod
    def kg_to_lb(kg: float) -> float:
        return kg / 0.453592