import streamlit as st


if 'inputs' not in st.session_state:
    st.session_state.inputs = set()

input_string = st.text_input("Give inputs")
st.session_state.inputs.add(input_string)

st.selectbox("Select Dynamic", options=list(st.session_state.inputs))


def main():
    # add the key choices_len to the session_state
    if not "choices_len" in st.session_state:
        st.session_state["choices_len"] = 0

    # c_up contains the form
    # c_down contains the add and remove buttons
    c_up = st.container()
    c_down = st.container()

    with c_up:
        with st.form("myForm"):
            c1 = st.container() # c1 contains choices
            c2 = st.container() # c2 contains submit button
            with c2:
                st.form_submit_button("submit")

    with c_down:
        col_l, _, col_r = st.columns((4,15,4))
        with col_l:
            if st.button("Add Choise"):
                st.session_state["choices_len"] += 1

        with col_r:
            if st.button("remove Choise") and st.session_state["choices_len"] > 1:
                st.session_state["choices_len"] -= 1
                st.session_state.pop(f'{st.session_state["choices_len"]}')


    for x in range(st.session_state["choices_len"]): # create many choices
        with c1:
            st.text_input("text", key=f"{x}")

    # reads values from the session_state using the key.
    # also doesn't return an option if the value is empty
    st.selectbox("myOptions", options=[
        st.session_state[f"{x}"] \
        for x in range(st.session_state["choices_len"]) \
        if not st.session_state[f"{x}"] == ''])

