"""
intersection.py

This module defines the Intersection class, which represents a traffic intersection
with optional traffic light control. It includes attributes for the intersection's
unique identifier, name, geographical location, and traffic light status.

Classes:
    Intersection: Represents a traffic intersection with optional traffic light control.

Dependencies:
    shapely.geometry.Point: Used to represent the geographical location of the intersection.

"""
from shapely.geometry import Point as POINT
from src.preprocessing.link import Link

class Intersection:
    """
    Represents a traffic intersection with optional traffic light control.

    Attributes:
        id (int): The unique identifier for the intersection.
        name (str): The name of the intersection.
        location (POINT): The geographical location of the intersection.
        tl (bool): Indicates whether the intersection has a traffic light. Defaults to True.

    Methods:
        __repr__(): Returns a string representation of the Intersection object.
    """
    def __init__(self, intersection_id: int, location: POINT, tl: bool = True,  link: Link = None):
        self.id = intersection_id
        self.location = location
        self.tl = tl
        self.link = link

    def set_link(self, link: Link):
        """
        Sets the link associated with the intersection.

        Args:
            link (Link): The link to be associated with the intersection.
        """
        if not isinstance(link, Link):
            raise TypeError("link must be an instance of Link")
        self.link = link

    def is_tl(self):
        """
        Returns whether the intersection has a traffic light.

        Returns:
            bool: True if the intersection has a traffic light, False otherwise.
        """
        return self.tl


    def get_link(self):
        """
        Returns the link associated with the intersection.
        """
        return self.link

    def __repr__(self):
        return f"Intersection(id={self.id}"

    def get_distance(self, point: POINT) -> float:
        """
        Returns the distance from the intersection to a given point.

        Args:
            point (POINT): The point to calculate the distance to.

        Returns:
            float: The distance from the intersection to the point.
        """
        # Ensure the point is within valid latitude and longitude ranges
        if point.x > 180 or point.x < -180 or point.y > 90 or point.y < -90:
            raise ValueError("Point must be within valid latitude and longitude ranges.")
        return self.location.distance(point)
