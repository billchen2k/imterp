import {MapView} from '@/components/map-view';
import {Sidebar} from '@/components/sidebar';
import {ThemeProvider, createTheme} from '@mui/material';
import {MapProvider} from 'react-map-gl';


const theme = createTheme({
  shape: {
    borderRadius: 3,
  },
  palette: {
    primary: {
      main: '#333333',
    },
  },
  typography: {
    fontFamily: 'Inconsolata',
  },
  components: {
    MuiFormControl: {
      defaultProps: {
        size: 'small',
        variant: 'standard',
      },
    },
    MuiButtonBase: {
      defaultProps: {
        disableRipple: true,
      },
    },
    MuiButton: {
      defaultProps: {
        variant: 'outlined',
        size: 'small',
      },
      styleOverrides: {
        root: {
          fontFamily: 'Helvetica Neue',
        },
      },
    },
    MuiSelect: {
      defaultProps: {
        size: 'small',
        variant: 'standard',
      },
    },
    MuiTextField: {
      defaultProps: {
        size: 'small',
        variant: 'standard',
      },
    },
  },
});

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <MapProvider>
        <div className={'flex flex-row w-screen h-screen max-w-[100vw]'}>
          <div className={'flex h-full w-[30rem]'}>
            <Sidebar />
          </div>
          <div className={'flex h-full flex-1 overflow-y-scroll'}>
            <MapView />
          </div>
        </div>
      </MapProvider>
    </ThemeProvider>

  );
};

export default App;
