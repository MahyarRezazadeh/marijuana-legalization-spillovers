clear all
mata:
lat1 =  40.0; lon1 = -86.0
lat2 =  37.5; lon2 = -119.5
pi = pi()
dlat = (lat2 - lat1) * pi / 180
dlon = (lon2 - lon1) * pi / 180
printf("dlat = %10.7f rad\n", dlat)
printf("dlon = %10.7f rad\n", dlon)
t1 = sin(dlat/2)
printf("sin(dlat/2) = %10.7f\n", t1)
t1sq = t1^2
printf("sin(dlat/2)^2 = %10.7f\n", t1sq)
cl1 = cos(lat1*pi/180)
cl2 = cos(lat2*pi/180)
printf("cos(lat1*pi/180) = %10.7f\n", cl1)
printf("cos(lat2*pi/180) = %10.7f\n", cl2)
t2 = sin(dlon/2)
printf("sin(dlon/2) = %10.7f\n", t2)
t2sq = t2^2
printf("sin(dlon/2)^2 = %10.7f\n", t2sq)
a = t1sq + cl1 * cl2 * t2sq
printf("a = %10.7f\n", a)
c = 2 * atan2(sqrt(a), sqrt(1-a))
printf("c = %10.7f rad\n", c)
printf("dist = %10.3f km\n", 6371*c)
printf("dist = %10.3f miles\n", 6371*c*0.621371)
// Now test with a known-good atan2
printf("atan2(0.225, 0.974) = %10.7f (expect ~0.227)\n", atan2(0.225, 0.974))
printf("atan2(0.974, 0.225) = %10.7f (expect ~1.344)\n", atan2(0.974, 0.225))
end
