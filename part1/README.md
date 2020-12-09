# 2020/11/19

* Creat material class and move brdf, pdf, sampling brdf functions into it
* Simplify pathtracer structure
   * raytrace function
      * Direct Intensity （if NEE on)
      * Next Intensity
         * Reflact intensity
         * Refract intensity
* Add transmission support
   * Stack overflow may happen
      * If shoot both reflact & refract rays at same time + NEE on + RR on -> overflow
         * Only when NEE is off, support shoot two rays
      * MIS + NEE on -> over flow
   * When NEE is on, too bright. When NEE off, edges not clear. Not sure the reason
   * General color distribution on the sphere seems reasonable