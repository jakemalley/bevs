import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

DATA_FILE = "2025.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df


df = load_data()

## Page Config
st.set_page_config(page_title="🍻 2025 Bevs", layout="wide")
st.title("🍻 2025 Bevs")

## Sidebar
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Filter Date Range", [df["Date"].min(), df["Date"].max()]
)

FLEAS = ["Jake", "Daniel", "Tom", "Alok", "James"]
exclude_fleas = st.sidebar.toggle("🦗 Exclude Fleas")
available_people = df["Person"].unique()
people = st.sidebar.multiselect(
    "Filter by Person", sorted(available_people), default=available_people
)

# Apply filters to the dataframe
filtered_df = df[
    (df["Date"].dt.date >= date_range[0])
    & (df["Date"].dt.date <= date_range[1])
    & (df["Person"].isin(people))
]

# Exclude fleas
if exclude_fleas:
    filtered_df = filtered_df[~filtered_df["Person"].isin(FLEAS)]

## Main Page
tab_analysis, tab_flea_swatter, tab_forecast = st.tabs(
    ["📊 2025 Analysis", "🦗 Flea Swatter", "🔮 Forecast"]
)

## Analysis
with tab_analysis:
    # High level stats
    st.subheader("🧐 High Level Stats")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Drinks", len(filtered_df))
    c2.metric("Unique People", filtered_df["Person"].nunique())
    c3.metric(
        "Top Drinker",
        filtered_df["Person"].value_counts().idxmax()
        if not filtered_df.empty
        else "N/A",
    )
    c4.metric(
        "Heaviest Day",
        filtered_df.groupby("Date").size().idxmax().strftime("%Y-%m-%d")
        if not filtered_df.empty
        else "N/A",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Location", filtered_df.groupby("Location").size().idxmax())
    c2.metric("Worst Location", filtered_df.groupby("Location").size().idxmin())
    c3.metric("Best Drink", filtered_df.groupby("Type").size().idxmax())
    c4.metric("Worst Drink", filtered_df.groupby("Type").size().idxmin())

    ## Leaderboard
    st.divider()
    st.subheader("🏆 Overall Leaderboard")

    overall = (
        filtered_df.groupby("Person")
        .size()
        .reset_index(name="Drinks")
        .sort_values("Drinks", ascending=False)
    )

    fig_overall = px.bar(
        overall,
        x="Person",
        y="Drinks",
        color="Person",
        text="Drinks",
        title="Total Drinks per Person",
    )
    fig_overall.update_traces(textposition="outside")
    fig_overall.update_layout(yaxis_title="Total Drinks")
    st.plotly_chart(fig_overall, width="stretch")

    ## Breakdown
    st.divider()
    st.subheader("🥧 Breakdown")

    c1, c2 = st.columns(2)
    with c1:
        fig_overall_pie = px.pie(
            overall,
            names="Person",
            values="Drinks",
            title="Breakdown per Person",
            hole=0.4,
        )
        fig_overall_pie.update_traces(textinfo="label", textposition="inside")
        st.plotly_chart(fig_overall_pie, width="stretch")

    with c2:
        fig_pie_by_type = px.pie(
            filtered_df.groupby("Type")
            .size()
            .reset_index(name="Drinks")
            .sort_values("Drinks", ascending=False),
            names="Type",
            values="Drinks",
            title="Breakdown by Type",
            hole=0.4,
        )

        fig_pie_by_type.update_traces(textinfo="label", textposition="inside")
        st.plotly_chart(fig_pie_by_type, width="stretch")

    ## Leaderboard (Weekly)
    st.divider()
    st.subheader("🏆 Weekly Leaderboard")

    iso = filtered_df["Date"].dt.isocalendar()

    weekly = (
        filtered_df.assign(Year=iso.year.astype(int), Week=iso.week.astype(int))
        .groupby(["Year", "Week", "Person"])
        .size()
        .reset_index(name="Drinks")
        .sort_values(["Year", "Week", "Drinks"], ascending=[True, True, False])
    )

    weekly["Year-Week"] = (
        weekly["Year"].astype(str) + "-W" + weekly["Week"].astype(str).str.zfill(2)
    )

    fig_weekly = px.bar(
        weekly,
        x="Person",
        y="Drinks",
        color="Person",
        animation_frame="Year-Week",
        title="Weekly Drinks Leaderboard",
    )

    st.plotly_chart(fig_weekly, width="stretch")

    ## Cumulative
    st.divider()
    st.subheader("📈 Cum(💦)ulative Drinks Over Time")

    cumulative = (
        filtered_df.groupby("Date")
        .size()
        .sort_index()
        .cumsum()
        .reset_index(name="Cumulative Drinks")
    )

    fig_cum = px.line(
        cumulative,
        x="Date",
        y="Cumulative Drinks",
        markers=True,
        title="Cum(💦)ulative Drinks Consumed",
    )

    st.plotly_chart(fig_cum, width="stretch")

    ## Daily Cumulative
    st.divider()
    st.subheader("📈 Cum(💦)ulative Daily Drinking")

    # Count per person per day
    pppd = filtered_df.groupby(["Person", "Date"]).size().rename("Cumulative Drinks")

    # Build full index (all persons × all dates)
    pppd_full_index = pd.MultiIndex.from_product(
        [
            pppd.index.get_level_values("Person").unique(),
            pd.date_range(
                pppd.index.get_level_values("Date").min(),
                pppd.index.get_level_values("Date").max(),
                freq="D",
            ),
        ],
        names=["Person", "Date"],
    )

    # Reindex and fill missing days
    pppd = (
        pppd.reindex(pppd_full_index, fill_value=0)
        .groupby(level="Person")
        .cumsum()
        .reset_index()
    )

    st.plotly_chart(
        px.bar(
            pppd,
            x="Person",
            y="Cumulative Drinks",
            color="Person",
            animation_frame="Date",
            title="Cum(💦)ulative Drinks",
        ),
        width="stretch",
    )

    ## Heatmap
    st.divider()
    st.subheader("📅 Calendar Heatmap")

    daily = (
        filtered_df.groupby(filtered_df["Date"].dt.date)
        .size()
        .reset_index(name="Drinks")
    )

    daily["date"] = pd.to_datetime(daily["Date"])

    iso = daily["date"].dt.isocalendar()
    daily["year"] = iso.year.astype(int)
    daily["week"] = iso.week.astype(int)
    daily["weekday"] = daily["date"].dt.weekday

    daily["year_week"] = (
        daily["year"].astype(str) + "-W" + daily["week"].astype(str).str.zfill(2)
    )

    fig_calendar = px.density_heatmap(
        daily,
        x="weekday",
        y="year_week",
        z="Drinks",
        color_continuous_scale="Viridis",
        labels={"weekday": "Day of Week", "year_week": "ISO Week", "Drinks": "Drinks"},
        title="Calendar Heatmap (Daily Drinks)",
    )

    fig_calendar.update_xaxes(
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )

    st.plotly_chart(fig_calendar, width="stretch")

    ## Worst 5 Drinkers
    st.divider()
    st.subheader("🥉 Worst 5 Drinkers")
    st.dataframe(
        filtered_df.groupby("Person")
        .size()
        .sort_values(ascending=True)
        .head(5)
        .reset_index(name="Drinks"),
        use_container_width=True,
        hide_index=True,
    )

    ## Raw Data
    st.divider()
    st.subheader("🥩 Filtered Raw Data")
    st.dataframe(filtered_df.sort_values("Date"), width="stretch")

## Flea Swatter
with tab_flea_swatter:
    st.subheader("🦗 Should we get rid of the fleas?")
    df_flea = df[
        (df["Date"].dt.date >= date_range[0])
        & (df["Date"].dt.date <= date_range[1])
        & (df["Person"].isin(FLEAS))
    ]

    df_other = df[
        (df["Date"].dt.date >= date_range[0])
        & (df["Date"].dt.date <= date_range[1])
        & ~(df["Person"].isin(FLEAS))
    ]

    def daily_cumulative(df, label, date_index):
        daily = (
            df.groupby("Date")
            .size()
            .reindex(date_index, fill_value=0)
            .cumsum()
            .reset_index(name="Cumulative Drinks")
        )
        daily["Group"] = label
        return daily

    date_index = pd.date_range(
        start=date_range[0],
        end=date_range[1],
        freq="D",
    )

    cum_fleas = daily_cumulative(df_flea, "Fleas", date_index)
    cum_other = daily_cumulative(df_other, "Others", date_index)
    cum_all = pd.concat([cum_fleas, cum_other], ignore_index=True)

    fig = px.line(
        cum_all,
        x="index",
        y="Cumulative Drinks",
        color="Group",
        markers=True,
        title="📈 Cumulative Drinks — Fleas vs Others",
        labels={"index": "Date"},
    )

    st.plotly_chart(fig, width="stretch")

    cumulative = (
        df_other.groupby("Date")
        .size()
        .sort_index()
        .reindex(
            pd.date_range(df["Date"].min(), df["Date"].max(), freq="D"),
            fill_value=0,
        )
        .cumsum()
    )

    prophet_df = pd.DataFrame(
        {
            "ds": cumulative.index,
            "y": cumulative.values,
        }
    )

    model = Prophet(
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_prior_scale=5.0,
        changepoint_prior_scale=0.05,
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=50)
    forecast = model.predict(future)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    TARGET = 2025

    hit = forecast[forecast["yhat"] >= TARGET]

    if hit.empty:
        st.warning("Forecast does not reach 2025 drinks in the forecast horizon.")
    else:
        hit_date = hit.iloc[0]["ds"]
        hit_value = hit.iloc[0]["yhat"]

        st.error(
            f"🍺 **Without fleas**, 2025 total drinks are forecast to be reached on "
            f"**{hit_date.date()}** (≈ {int(hit_value)} drinks)"
        )

    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="🔮 Non-Flea Cumulative Drinks Forecast",
        labels={"ds": "Date", "yhat": "Cumulative Drinks"},
    )

    fig.add_hline(
        y=TARGET,
        line_dash="dash",
        line_color="red",
        annotation_text="2025 drinks",
        annotation_position="top left",
    )

    st.plotly_chart(fig, use_container_width=True)

## Actual Forecast
with tab_forecast:
    st.subheader("🔮 Forecast Cumulative Drinks per Person into 2026")

    st.markdown("""
    Forecast of cumulative drink counts per person using **Meta Prophet**.
    Each person's drinking trajectory is modeled separately and projected forward.
    """)

    # Person selector
    selected_people = st.multiselect(
        "Select people to forecast",
        available_people,
        default=FLEAS,
    )

    if not selected_people:
        st.warning("Please select at least one person to forecast.")
    else:
        # Forecast parameters
        forecast_days = st.slider("Days to forecast", 30, 365, 180)

        st.divider()

        # -----------------------------
        # Prepare and forecast per person
        # -----------------------------
        all_forecasts = []
        all_actuals = []

        for person in selected_people:
            # Get person's drink history
            person_df = df[df["Person"] == person].copy()

            # Daily counts
            daily_counts = (
                person_df.groupby("Date")
                .size()
                .sort_index()
                .reindex(
                    pd.date_range(df["Date"].min(), df["Date"].max(), freq="D"),
                    fill_value=0,
                )
            )

            # Cumulative
            cumulative = daily_counts.cumsum()

            # Prepare for Prophet
            prophet_df = pd.DataFrame({"ds": cumulative.index, "y": cumulative.values})

            # Store actuals for plotting
            actual_df = prophet_df.copy()
            actual_df["Person"] = person
            all_actuals.append(actual_df)

            # Train Prophet
            model = Prophet(
                growth="linear",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_prior_scale=5.0,
                changepoint_prior_scale=0.05,
            )

            model.fit(prophet_df)

            # Forecast
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            # Ensure non-negative and monotonic (cumulative must increase)
            forecast["yhat"] = forecast["yhat"].clip(lower=0)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

            # Add person identifier
            forecast["Person"] = person
            all_forecasts.append(forecast)

        # Combine all forecasts
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        combined_actuals = pd.concat(all_actuals, ignore_index=True)

        # Plot forecast
        fig = px.line(
            combined_forecast,
            x="ds",
            y="yhat",
            color="Person",
            title="Forecasted Cumulative Drinks per Person",
            labels={"ds": "Date", "yhat": "Cumulative Drinks", "Person": "Person"},
        )

        # Add actual historical data as scatter
        for person in selected_people:
            person_actual = combined_actuals[combined_actuals["Person"] == person]
            fig.add_scatter(
                x=person_actual["ds"],
                y=person_actual["y"],
                mode="markers",
                name=f"{person} (actual)",
                marker=dict(size=4),
                showlegend=True,
            )

        fig.update_layout(hovermode="x unified", height=600)

        st.plotly_chart(fig, width="stretch")

        st.divider()

        # Projected totals table
        st.subheader("📊 Projected Totals")

        end_date = df["Date"].max() + pd.Timedelta(days=forecast_days)

        projections = []
        for person in selected_people:
            person_forecast = combined_forecast[combined_forecast["Person"] == person]

            current_total = combined_actuals[combined_actuals["Person"] == person][
                "y"
            ].iloc[-1]
            projected_total = person_forecast[person_forecast["ds"] == end_date][
                "yhat"
            ].values[0]
            projected_increase = projected_total - current_total

            projections.append(
                {
                    "Person": person,
                    "Current Total": int(current_total),
                    "Projected Total": int(projected_total),
                    "Projected Increase": int(projected_increase),
                    "Avg Drinks/Day (projected)": round(
                        projected_increase / forecast_days, 2
                    ),
                }
            )

        projection_df = pd.DataFrame(projections).sort_values(
            "Projected Total", ascending=False
        )

        st.dataframe(projection_df, width="stretch", hide_index=True)
