from bs4 import BeautifulSoup as BS
from datetime import datetime as dt
from numpy import pi, sin, cos, arcsin, sqrt, arctan2
import numpy as np
import sys

def earth_radius(lat):
    """
    Calculates the Earth's radius at the given latitude assuming it
    varies linearly with cos(latitude). Latitude must be given in
    radians. Return value is in km.
    """

    return 6356.752 + (6378.137 - 6356.752) * cos(lat)

def distance(p1, p2):
    """
    Calculates the distance between TrackingPoint objects p1 and p2
    using the Haversine formula
    (https://en.wikipedia.org/wiki/Haversine_formula). This method
    assumes Earth is a perfect sphere, so it is accurate to no better
    than 0.5%. That's good enough for me.
    """

    lat_1 = p1.get_latitude(radians = True)
    lat_2 = p2.get_latitude(radians = True)

    R_E = earth_radius(lat_1) * 1000 # Earth's radius in m

    lon_1 = p1.get_longitude(radians = True)
    lon_2 = p2.get_longitude(radians = True)

    t1 = sin((lat_2 - lat_1) / 2)**2
    t2 = cos(lat_1) * cos(lat_2) * sin((lon_2 - lon_1) / 2)**2
    return 2 * R_E * arcsin(sqrt(t1 + t2))

def bearing(p1, p2):
    """
    Calculates the bearing to follow from TrackingPoint p1 to
    TrackingPoint p2. Formula from
    https://www.movable-type.co.uk/scripts/latlong.html.
    """
    lat_1 = p1.get_latitude(radians = True)
    lat_2 = p2.get_latitude(radians = True)

    lon_1 = p1.get_longitude(radians = True)
    lon_2 = p2.get_longitude(radians = True)

    t1 = cos(lat_1) * sin(lat_2)
    t2 = sin(lat_1) * cos(lat_2) * cos(lon_2 - lon_1)

    f1 = t1 - t2
    f2 = sin(lon_2 - lon_1) * cos(lat_2)

    return arctan2(f2, f1)

def displacement(p1, p2):
    """
    Calculates the displacement between TrackingPoints p1 and p2 in
    meters. Returned value is [meters East, meters North].
    """

    dir = bearing(p1, p2)
    dist = distance(p1, p2)

    return np.array([dist * sin(dir), dist * cos(dir)])

class TrackingPoint(object):
    """
    Class to tidy up the <trkpt> objects in .gpx files. Information on
    the filetype can be found at
    http://www.topografix.com/GPX/1/1/gpx.xsd
    Information on gpx extensions can be found at
    http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd
    and at
    http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd
    """

    def __init__(self, trkpt):
        """
        Strips information from the <trkpt> item found in gpx files.
        The info I've decided to strip is what shows up in my gpx
        files. Heart rate will not be present if you don't have an HR
        monitor. Position info may not be as accurate if you don't
        use a GPS.
        """

        # Latitude and longitude in decimal degrees
        self.latitude = float(trkpt['lat'])
        self.longitude = float(trkpt['lon'])

        # Elevation in meters
        self.elevation = float(trkpt.ele.text)

        # Heart rate in BPM
        self.hr_bpm = trkpt.extensions.TrackPointExtension.hr.text

        try:
            self.hr_bpm = int(self.hr_bpm)
        except: # If heartrate is not included in the data
            self.hr_bpm = -1

        # Time in UTC (not local)
        self.time = dt.strptime(trkpt.time.text, "%Y-%m-%dT%H:%M:%SZ")

    def __repr__(self):
        return "TrackingPoint(lat=%r,long=%r,hr=%r,ele=%r," \
               "time=%r)" % (self.latitude, self.longitude,
                             self.hr_bpm, self.elevation, self.time)

    # Getters
    def get_latitude(self, radians = False):
        if radians:
            return np.radians(self.latitude)

        return self.latitude

    def get_longitude(self, radians = False):
        if radians:
            return np.radians(self.longitude)

        return self.longitude

    def get_hr(self):
        return self.hr_bpm

    def get_elevation(self):
        return self.elevation

    def get_time(self):
        return self.time

class BikeRide(object):
    """
    Obtains data from a .gpx file and arranges it into a readable
    format. Some of the data I collect here is specific to my gear
    (heart rate, latitude, longitude, etc.).
    """

    def __init__(self, gpx_file):
        """
        Creates a list of TrackingPoint objects for every point along
        the ride located in gpx_file.
        """

        raw_data = self.load_bike_data(gpx_file)
        if raw_data == None:
            raise FileNotFoundError("Invalid file passed to " \
                                    "BikeRide.__init__")

        self.tracking_points = map(TrackingPoint,
                                   raw_data.find_all('trkpt'))
        self.tracking_points = np.array(list(self.tracking_points))
        self.start = self.tracking_points[0]
        self.end = self.tracking_points[-1]
        self.num_points = len(self.tracking_points)
        self.path = self.calculate_path()

    def __repr__(self):
        return "BikeRide(start=%r,end=%r,num_points=%r)" % \
                                                (self.start,
                                                 self.end,
                                                 self.num_points)

    def load_bike_data(self, gpx_file):
        """
        Import using BeautifulSoup.
        """
        try:
            with open(gpx_file) as gp:
                soup = BS(gp, 'xml')
        except FileNotFoundError:
            return None

        return soup

    def calculate_path(self):
        """
        Returns the path in meters, assuming tracking_points[0] is
        located at (0, 0).
        """

        positions = np.array([np.array([0., 0.])])
        current_position = np.array([0., 0.])

        for index in range(1, len(self.tracking_points)):
            point = self.tracking_points[index]
            previous = self.tracking_points[index - 1]
            current_position += displacement(previous, point)
            positions = np.append(positions,
                                  np.array([current_position]),
                                  axis = 0)

        return positions

    def get_path(self):
        return self.path
