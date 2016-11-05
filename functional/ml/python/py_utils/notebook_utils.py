

def get_code_toggle(on_by_default=True, button_label='Toggle Code'):
    from IPython.display import HTML
    default_value = 'true' if on_by_default else 'false'
    return HTML(
        """
        <script>
        code_show="""+default_value+""";
        function code_toggle() {
         if (code_show){
         $('div.input').hide();
         } else {
         $('div.input').show();
         }
         code_show = !code_show
        }
        $( document ).ready(code_toggle);
        </script>
        <form action="javascript:code_toggle()">
        <input type="submit" value=" """ + button_label + """ "></form>
        """
    )
