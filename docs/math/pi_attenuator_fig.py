import schemdraw
import schemdraw.elements as elm

with schemdraw.Drawing():
    elm.Resistor().down().hold()
    # elm.Ground(lead=False)
    elm.Resistor().right()
    elm.Resistor().down()
    elm.Ground(lead=False)

# Resistor labels
R_shunt1 = 'R₁'
R_series = 'R₂'
R_shunt2 = 'R₃'


with schemdraw.Drawing() as d:
    d.config(unit=2.5)

    # First shunt resistor (left leg of the π)
    d.push()
    d += elm.Resistor().down().label(R_shunt1, loc='bottom')
    d += elm.Ground()
    d.pop()

    # Series resistor (middle bar of the π)
    d += elm.Resistor().right().label(R_series)

    # Second shunt resistor (right leg of the π)
    d.push()
    d += elm.Resistor().down().label(R_shunt2, loc='bottom')
    d += elm.Ground()
    d.pop()

    d.save('pi_attenuator_fig.pdf')
