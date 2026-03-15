"""Pandas DataFrame accessor for splita.

Registers a ``.splita`` accessor on ``pandas.DataFrame`` when imported::

    import splita.integrations.pandas_accessor

    df.splita.experiment("control", "treatment")
    df.splita.check_srm("group")

Lazy import --- only registers when the user explicitly imports this module.
"""

from __future__ import annotations

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("splita")
class SplitaAccessor:
    """Pandas DataFrame accessor exposing splita analysis methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame this accessor is attached to.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def experiment(
        self,
        control_col: str,
        treatment_col: str,
        **kwargs: object,
    ) -> object:
        """Run an A/B test on two DataFrame columns.

        Parameters
        ----------
        control_col : str
            Column name for the control group observations.
        treatment_col : str
            Column name for the treatment group observations.
        **kwargs
            Additional keyword arguments passed to
            :class:`~splita.Experiment`.

        Returns
        -------
        ExperimentResult
            The experiment result dataclass.

        Raises
        ------
        KeyError
            If either column is not in the DataFrame.
        """
        from splita import Experiment

        if control_col not in self._df.columns:
            raise KeyError(
                f"Column {control_col!r} not found in DataFrame. "
                f"Available columns: {list(self._df.columns)}"
            )
        if treatment_col not in self._df.columns:
            raise KeyError(
                f"Column {treatment_col!r} not found in DataFrame. "
                f"Available columns: {list(self._df.columns)}"
            )

        ctrl = self._df[control_col].dropna().values
        trt = self._df[treatment_col].dropna().values
        return Experiment(ctrl, trt, **kwargs).run()

    def check_srm(
        self,
        group_col: str,
        *,
        expected_fractions: list[float] | None = None,
        **kwargs: object,
    ) -> object:
        """Run an SRM check on group assignment counts.

        Parameters
        ----------
        group_col : str
            Column name containing group labels (e.g. "A", "B").
        expected_fractions : list of float or None, default None
            Expected traffic fraction per variant. If None, equal split
            is assumed.
        **kwargs
            Additional keyword arguments passed to
            :class:`~splita.SRMCheck`.

        Returns
        -------
        SRMResult
            The SRM check result dataclass.

        Raises
        ------
        KeyError
            If the column is not in the DataFrame.
        """
        from splita import SRMCheck

        if group_col not in self._df.columns:
            raise KeyError(
                f"Column {group_col!r} not found in DataFrame. "
                f"Available columns: {list(self._df.columns)}"
            )

        counts = self._df[group_col].value_counts().sort_index().tolist()
        return SRMCheck(counts, expected_fractions=expected_fractions, **kwargs).run()
