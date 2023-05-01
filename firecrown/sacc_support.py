"""Adds support for new data types to sacc."""

from importlib import reload

from astropy.table import Table

import sacc
from sacc.utils import Namespace
from sacc.tracers import BaseTracer

sacc.data_types.required_tags["cluster_counts"] = []
sacc.data_types.required_tags["cluster_mean_mass"] = []

sacc.data_types.standard_types = Namespace(*sacc.data_types.required_tags.keys())


class BinZTracer(BaseTracer, tracer_type="bin_z"):  # type: ignore
    """A tracer for a single redshift bin."""

    def __init__(self, name: str, z_lower: float, z_upper: float, **kwargs):
        """
        Create a tracer corresponding to a single redshift bin.

        :param name: The name of the tracer
        :param z_lower: The lower bound of the redshift bin
        :param z_upper: The upper bound of the redshift bin
        """
        super().__init__(name, **kwargs)
        self.z_lower = z_lower
        self.z_upper = z_upper

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of BinZTracers to a single astropy table

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List with a single astropy table
        """

        names = ["name", "quantity", "z_lower", "z_upper"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.z_lower for obj in instance_list],
            [obj.z_upper for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.tracer_type
        table.meta["EXTNAME"] = f"tracer:{cls.tracer_type}"
        return [table]

    @classmethod
    def from_tables(cls, table_list):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        A single tracer object is read from the table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for table in table_list:
            for row in table:
                name = row["name"]
                quantity = row["quantity"]
                z_lower = row["z_lower"]
                z_upper = row["z_upper"]
                tracers[name] = cls(
                    name, quantity=quantity, z_lower=z_lower, z_upper=z_upper
                )
        return tracers


class BinRichnessTracer(BaseTracer, tracer_type="bin_richness"):  # type: ignore
    """A tracer for a single richness bin."""

    def __init__(
        self, name: str, richness_lower: float, richness_upper: float, **kwargs
    ):
        """
        Create a tracer corresponding to a single richness bin.

        :param name: The name of the tracer
        :param richness_lower: The lower bound of the redshift bin
        :param richness_upper: The upper bound of the redshift bin
        """
        super().__init__(name, **kwargs)
        self.richness_lower = richness_lower
        self.richness_upper = richness_upper

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of BinZTracers to a list of astropy tables

        This is used when saving data to a file.
        One table is generated per tracer.

        :param instance_list: List of tracer instances
        :return: List of astropy tables
        """
        names = ["name", "quantity", "richness_lower", "richness_upper"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.richness_lower for obj in instance_list],
            [obj.richness_upper for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.tracer_type
        table.meta["EXTNAME"] = f"tracer:{cls.tracer_type}"
        return [table]

    @classmethod
    def from_tables(cls, table_list):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        A single tracer object is read from the table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for table in table_list:
            for row in table:
                name = row["name"]
                quantity = row["quantity"]
                richness_lower = row["richness_lower"]
                richness_upper = row["richness_upper"]
                tracers[name] = cls(
                    name,
                    quantity=quantity,
                    richness_lower=richness_lower,
                    richness_upper=richness_upper,
                )
        return tracers


class ClusterSurveyTracer(BaseTracer, tracer_type="cluster_survey"):  # type: ignore
    """A tracer for a single richness bin."""

    def __init__(self, name: str, sky_area: float, **kwargs):
        """
        Create a tracer corresponding to a single richness bin.

        :param name: The name of the tracer
        :param sky_area: The survey's sky area in square degrees
        """
        super().__init__(name, **kwargs)
        self.sky_area = sky_area

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of BinZTracers to a list of astropy tables

        This is used when saving data to a file.
        One table is generated per tracer.

        :param instance_list: List of tracer instances
        :return: List of astropy tables
        """
        names = ["name", "quantity", "sky_area"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.sky_area for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.tracer_type
        table.meta["EXTNAME"] = f"tracer:{cls.tracer_type}"
        return [table]

    @classmethod
    def from_tables(cls, table_list):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        A single tracer object is read from the table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for table in table_list:
            for row in table:
                name = row["name"]
                quantity = row["quantity"]
                sky_area = row["sky_area"]
                tracers[name] = cls(
                    name,
                    quantity=quantity,
                    sky_area=sky_area,
                )
        return tracers


reload(sacc)