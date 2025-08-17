{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

``{{ fullname }}``

.. autoclass:: {{ objname }}
   :show-inheritance:
   :class-doc-from: both

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      {% if not item in skipmethods and not item in inherited_members %}
          {{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes  %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {% if not item in inherited_members %}
         {{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
